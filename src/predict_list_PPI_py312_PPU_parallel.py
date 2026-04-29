#!/usr/local/bin/python
import os
import sys
import time
import string
import argparse
import shutil
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from data_loader import get_msa, MSAFeaturize
from RoseTTAFoldModel import RF2trackModule
from kinematics import xyz_to_t2d
from chemical import INIT_CRDS
from AuxiliaryPredictor import DistanceNetwork


N_cycle = 3


def get_args():
    parser = argparse.ArgumentParser(
        description="RF2-PPI parallel inference for ZW810E/PPU."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Input list file. Each line: paired_msa.a3m<TAB>L1"
    )
    parser.add_argument(
        "-list_fn",
        default=None,
        help="Input list file. Same as positional input_file."
    )
    parser.add_argument(
        "-model_file",
        default=None,
        help="RF2-PPI.pt path. Default: ./models/RF2-PPI.pt or ../models/RF2-PPI.pt"
    )
    parser.add_argument(
        "-number_seqs",
        type=int,
        default=5000,
        help="Number of paired sequences to include for inference. Default: 5000"
    )
    parser.add_argument(
        "--num_ppus",
        type=int,
        default=None,
        help="Number of PPU devices to use. Default: all visible PPU/CUDA-compatible devices."
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated device ids, e.g. 0,1,2,3. Overrides --num_ppus."
    )
    return parser.parse_args()


def resolve_input_file(args):
    input_fn = args.list_fn or args.input_file
    if not input_fn:
        raise ValueError("Please provide input file by positional argument or -list_fn.")
    if not os.path.exists(input_fn):
        raise FileNotFoundError(input_fn)
    return input_fn


def resolve_model_file(args):
    if args.model_file:
        if not os.path.exists(args.model_file):
            raise FileNotFoundError(args.model_file)
        return args.model_file

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "models" / "RF2-PPI.pt",
        script_dir.parent / "models" / "RF2-PPI.pt",
        Path.cwd() / "models" / "RF2-PPI.pt",
        Path.cwd().parent / "models" / "RF2-PPI.pt",
    ]

    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "RF2-PPI.pt not found. Please specify -model_file /path/to/RF2-PPI.pt"
    )


def ensure_ppu_env():
    os.environ.setdefault("PPU_HOME", "/usr/local/PPU_SDK")
    os.environ.setdefault("PPU_SDK", "/usr/local/PPU_SDK")

    lib_path = "/usr/local/PPU_SDK/targets/x86_64-linux/lib"
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_path not in old_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = lib_path + (":" + old_ld if old_ld else "")


def get_visible_devices(args):
    ensure_ppu_env()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA-compatible PPU device is available. CPU fallback is disabled. "
            "Check PPU_HOME/PPU_SDK/LD_LIBRARY_PATH and run with /usr/local/bin/python outside conda."
        )

    n = int(torch.cuda.device_count())
    if n <= 0:
        raise RuntimeError("torch.cuda.device_count() is 0. No visible PPU device.")

    if args.devices:
        devices = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    elif args.num_ppus is not None:
        devices = list(range(min(args.num_ppus, n)))
    else:
        devices = list(range(n))

    if not devices:
        raise RuntimeError("No devices selected.")

    for d in devices:
        if d < 0 or d >= n:
            raise ValueError(f"Invalid device id {d}; visible device count is {n}")

    return devices


def read_first_msa_sequence(msa_file):
    seq_lines = []
    with open(msa_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_lines:
                    break
                continue
            seq_lines.append(line)
    return "".join(seq_lines)


def estimate_token(msa_file):
    # Match inference behavior: lowercase insertion letters are removed before MSA encoding.
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    try:
        seq = read_first_msa_sequence(msa_file).translate(table)
        length = len(seq)
        if length <= 0:
            length = 1000
    except Exception:
        length = 1000
    return length * length


def load_done_from_result(result_fn):
    done = set()
    if not os.path.exists(result_fn):
        return done

    with open(result_fn) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Input_MSA"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                done.add(parts[0])
    return done


def load_tasks(input_fn, done_set):
    tasks = []
    with open(input_fn) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            words = line.split()
            if len(words) != 2:
                continue

            msa_file = words[0]
            if msa_file in done_set:
                continue

            try:
                L1 = int(words[1])
            except Exception:
                continue

            token = estimate_token(msa_file)
            tasks.append({"msa_file": msa_file, "L1": L1, "token": token, "line": line})
    return tasks


def split_tasks_by_token(tasks, devices):
    tasks = sorted(tasks, key=lambda x: x["token"], reverse=True)
    buckets = {d: [] for d in devices}
    bucket_tokens = {d: 0 for d in devices}

    for task in tasks:
        dev = min(devices, key=lambda x: bucket_tokens[x])
        buckets[dev].append(task)
        bucket_tokens[dev] += task["token"]

    print("TOTAL REMAINING TASKS:", len(tasks), flush=True)
    print("", flush=True)
    print("[TOKEN BALANCE]", flush=True)
    for d in devices:
        print(f"PPU{d}: {len(buckets[d])} tasks | tokens={bucket_tokens[d]}", flush=True)
    print("", flush=True)
    return buckets, bucket_tokens


def prep_input(msa, L1, L2, device):
    msa_in = torch.tensor(msa)
    msa_orig = msa_in[None].long().to(device)  # (B, N, L)

    idx = torch.arange(L1 + L2)
    idx[L1:] += 200
    idx = idx[None].to(device)

    chain_idx = torch.zeros((L1 + L2, L1 + L2)).long()
    chain_idx[:L1, :L1] = 1
    chain_idx[L1:, L1:] = 1
    chain_idx = chain_idx[None].to(device)

    qlen = L1 + L2
    xyz_t = INIT_CRDS.reshape(1, 1, 27, 3).repeat(1, qlen, 1, 1)[None].to(device)

    t1d = torch.nn.functional.one_hot(
        torch.full((1, qlen), 20, device=device).long(),
        num_classes=21
    ).float()

    conf = torch.zeros((1, qlen, 1), device=device).float()
    t1d = torch.cat((t1d, conf), -1)[None]

    mask_t = torch.full((1, qlen, 27), False, device=device)[None]

    mask_t_2d = mask_t[:, :, :, :3].all(dim=-1)
    mask_t_2d = mask_t_2d[:, :, None] * mask_t_2d[:, :, :, None]
    mask_t_2d = mask_t_2d.float() * chain_idx.float()[:, None]

    t2d = xyz_to_t2d(xyz_t, mask_t_2d)

    network_input = {}
    network_input["msa_orig"] = msa_orig
    network_input["idx"] = idx
    network_input["t1d"] = t1d
    network_input["t2d"] = t2d
    network_input["same_chain"] = chain_idx
    return network_input


def _get_model_input(network_input, output_i, return_raw=False, use_checkpoint=False):
    input_i = {}
    for key in network_input:
        if key == "msa_orig":
            seq, msa_seed, msa_extra, mask_msa, msa_seed_gt = MSAFeaturize(
                network_input["msa_orig"],
                {"MAXLAT": 128, "MAXSEQ": 1024},
                p_mask=-1.0
            )
            input_i["seq"] = seq
            input_i["msa_seed"] = msa_seed
            input_i["msa_extra"] = msa_extra
        else:
            input_i[key] = network_input[key]

    msa_prev, pair_prev = output_i
    input_i["msa_prev"] = msa_prev
    input_i["pair_prev"] = pair_prev
    input_i["return_raw"] = return_raw
    input_i["use_checkpoint"] = use_checkpoint
    return input_i


def read_msa_as_uint8(pair, num_seqs, table):
    msa = []
    seq_line = []
    with open(pair) as f_ent:
        for raw_line in f_ent:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            if raw_line[0] == ">":
                if seq_line:
                    msa.append("".join(seq_line).translate(table))
                    seq_line = []
                    if len(msa) >= num_seqs:
                        break
                continue

            seq_line.append(raw_line)

        if seq_line and len(msa) < num_seqs:
            msa.append("".join(seq_line).translate(table))

    if not msa:
        raise ValueError(f"No sequence found in MSA file: {pair}")

    arr = np.array([list(s) for s in msa], dtype="|S1").view(np.uint8)
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYVX-"), dtype="|S1").view(np.uint8)

    for i in range(alphabet.shape[0]):
        arr[arr == alphabet[i]] = i

    arr[arr > 21] = 21
    return arr, len(msa[0])


def is_oom_error(exc):
    msg = str(exc).lower()
    oom_keys = ["out of memory", "oom", "memory", "allocation", "alloc", "not enough", "hbm"]
    return any(k in msg for k in oom_keys)


def append_line_locked(lock, path, line):
    with lock:
        with open(path, "a") as f:
            f.write(line.rstrip("\n") + "\n")
            f.flush()


def worker(device_id, tasks, model_file, num_seqs, result_fn, failed_fn, tmp_dir, file_lock):
    ensure_ppu_env()

    # ZW810E/PPU exposes itself through CUDA-compatible APIs.
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"

    model = RF2trackModule().to(device)
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    local_results = {}

    for task in tasks:
        pair = task["msa_file"]
        L1 = task["L1"]

        try:
            msa, t_length = read_msa_as_uint8(pair, num_seqs, table)
            L2 = t_length - L1

            startT = time.time()
            network_input = prep_input(msa, L1, L2, device)

            with torch.no_grad():
                output_i = (None, None)
                for i_cycle in range(N_cycle):
                    return_raw = i_cycle < N_cycle - 1
                    input_i = _get_model_input(
                        network_input,
                        output_i,
                        return_raw=return_raw,
                        use_checkpoint=False
                    )
                    output_i = model(**input_i)

                    if i_cycle < N_cycle - 1:
                        continue

                    logit_dist = output_i[0][0]
                    prob_dist = torch.nn.Softmax(dim=1)(logit_dist)
                    p_bind = prob_dist[:, :20].sum(dim=1)
                    p_bind = p_bind * (1.0 - network_input["same_chain"].float())
                    p_bind = torch.nn.MaxPool2d(p_bind.shape[1:])(p_bind).view(-1)
                    p_bind = torch.clamp(p_bind, min=0.0, max=1.0)

                    prob = np.sum(
                        prob_dist[0]
                        .permute(1, 2, 0)
                        .detach()
                        .cpu()
                        .numpy()[:L1, L1:, :20],
                        axis=-1
                    ).astype(np.float16)

                    local_results[pair] = prob
                    endT = time.time()

                    append_line_locked(
                        file_lock,
                        result_fn,
                        pair + "\t" + str(round(p_bind.item(), 6)) + "\t" + str(round(endT - startT, 6))
                    )

        except RuntimeError as e:
            reason = "OOM" if is_oom_error(e) else "RuntimeError"
            append_line_locked(file_lock, failed_fn, f"{pair}\t{reason}\t{str(e).splitlines()[0]}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue

        except Exception as e:
            append_line_locked(file_lock, failed_fn, f"{pair}\tERROR\t{str(e).splitlines()[0]}")
            continue

    tmp_npz = os.path.join(tmp_dir, f"worker_ppu{device_id}.npz")
    np.savez_compressed(tmp_npz, **local_results)
    print(f"[PPU {device_id}] ALL TASKS DONE", flush=True)


def ensure_result_header(result_fn):
    if not os.path.exists(result_fn) or os.path.getsize(result_fn) == 0:
        with open(result_fn, "w") as f:
            f.write("Input_MSA\tInteraction_probability\tCompute_time\n")


def merge_npz(final_npz, tmp_dir):
    merged = {}

    if os.path.exists(final_npz):
        try:
            old = np.load(final_npz, allow_pickle=False)
            for k in old.files:
                merged[k] = old[k]
        except Exception:
            pass

    for p in sorted(Path(tmp_dir).glob("worker_ppu*.npz")):
        try:
            d = np.load(str(p), allow_pickle=False)
            for k in d.files:
                merged[k] = d[k]
        except Exception:
            continue

    np.savez_compressed(final_npz, **merged)


def count_valid_input_tasks(input_fn):
    total = 0
    with open(input_fn) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            words = line.split()
            if len(words) == 2:
                total += 1
    return total


def main():
    mp.set_start_method("spawn", force=True)
    args = get_args()

    input_fn = resolve_input_file(args)

    result_fn = input_fn + ".810E_rf2ppi.result"
    failed_fn = input_fn + ".failed"
    final_npz = input_fn + ".npz"
    tmp_dir = input_fn + ".810E_rf2ppi.tmp"

    # Step 1: check completed jobs first.
    # Only .810E_rf2ppi.result is used to define completed jobs.
    # .failed is not used for pre-filtering; OOM/failed jobs are detected during this run.
    ensure_result_header(result_fn)
    done_set = load_done_from_result(result_fn)
    total_input_tasks = count_valid_input_tasks(input_fn)
    tasks = load_tasks(input_fn, done_set)

    print(f"TOTAL INPUT TASKS: {total_input_tasks}", flush=True)
    print(f"COMPLETED TASKS IN RESULT: {len(done_set)}", flush=True)
    print(f"TOTAL REMAINING TASKS: {len(tasks)}", flush=True)

    if not tasks:
        print("ALL DONE", flush=True)
        return

    # Device/model checks and model loading happen only after remaining tasks are confirmed.
    model_file = resolve_model_file(args)
    devices = get_visible_devices(args)

    os.makedirs(tmp_dir, exist_ok=True)

    active_devices = devices[:min(len(devices), len(tasks))]
    buckets, bucket_tokens = split_tasks_by_token(tasks, active_devices)

    manager = mp.Manager()
    file_lock = manager.Lock()

    processes = []
    for device_id in active_devices:
        p = mp.Process(
            target=worker,
            args=(
                device_id,
                buckets[device_id],
                model_file,
                args.number_seqs,
                result_fn,
                failed_fn,
                tmp_dir,
                file_lock,
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    merge_npz(final_npz, tmp_dir)

    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
