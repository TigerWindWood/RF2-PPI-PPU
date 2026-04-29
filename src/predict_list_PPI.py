import os, sys, time, torch, string, zipfile, argparse
import numpy as np
from data_loader import get_msa, MSAFeaturize
from RoseTTAFoldModel  import RF2trackModule
from kinematics import xyz_to_t2d
from chemical import INIT_CRDS
from AuxiliaryPredictor import DistanceNetwork


N_cycle = 3
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-list_fn', required=True, help='List file containing input pMSA, L1 (length of the first protein)')
    parser.add_argument('-model_file', required=True, help='The trained model to use')
    parser.add_argument('-number_seqs', type=int, default=5000, help="Number of paired sequences to include for inference. Default is 5000")
    args = parser.parse_args()
    return args


def prep_input(msa, L1, L2, device):
    msa_in = torch.tensor(msa)
    msa_orig = msa_in[None].long().to(device) # (B, N, L)
    idx = torch.arange(L1 + L2)
    idx[L1:] += 200
    idx = idx[None].to(device)

    chain_idx = torch.zeros((L1 + L2, L1 + L2)).long()
    chain_idx[:L1, :L1] = 1
    chain_idx[L1:, L1:] = 1
    chain_idx = chain_idx[None].to(device)

    qlen = L1 + L2
    xyz_t = INIT_CRDS.reshape(1,1,27,3).repeat(1,qlen,1,1)[None].to(device) # (B, T, L, 27, 3)
    t1d = torch.nn.functional.one_hot(torch.full((1, qlen), 20, device=device).long(), num_classes=21).float() # all gaps
    conf = torch.zeros((1, qlen, 1), device=device).float()
    t1d = torch.cat((t1d, conf), -1)[None] # (B, T, L, D)
    mask_t = torch.full((1,qlen,27), False, device=device)[None] # (B, T, L, 27)

    mask_t_2d = mask_t[:,:,:,:3].all(dim=-1) # (B, T, L)
    mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
    mask_t_2d = mask_t_2d.float() * chain_idx.float()[:,None] # (ignore inter-chain region because we don't use any complex templates)
    t2d = xyz_to_t2d(xyz_t, mask_t_2d) # 2D features for templates (B, T, L, L, 37+6+1)

    network_input = {}
    network_input['msa_orig'] = msa_orig
    network_input['idx'] = idx
    network_input['t1d'] = t1d
    network_input['t2d'] = t2d
    network_input['same_chain'] = chain_idx
    return network_input


def _get_model_input(network_input, output_i, return_raw=False, use_checkpoint=False):
    '''
    Get network inputs for the current recycling step
    It will return dictionary of network inputs & labels for MLM tasks (mask_msa & msa_seed_gt)
    '''
    input_i = {}
    for key in network_input:
        if key == 'msa_orig':
            seq, msa_seed, msa_extra, mask_msa, msa_seed_gt = MSAFeaturize(network_input['msa_orig'], {'MAXLAT': 128, 'MAXSEQ':1024}, p_mask=-1.0)
            input_i['seq'] = seq # masked query sequence
            input_i['msa_seed'] = msa_seed # masked seed MSA
            input_i['msa_extra'] = msa_extra # extra MSA
        else:
            input_i[key] = network_input[key]
    msa_prev, pair_prev = output_i
    input_i['msa_prev'] = msa_prev
    input_i['pair_prev'] = pair_prev
    input_i['return_raw'] = return_raw
    input_i['use_checkpoint'] = use_checkpoint
    return input_i


def _rf2ppi_backend_available(torch_module, backend_name):
    """Safely check whether a torch accelerator backend is available."""
    backend = getattr(torch_module, backend_name, None)
    if backend is None:
        return False
    is_available = getattr(backend, "is_available", None)
    if is_available is None:
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def _rf2ppi_backend_device_count(torch_module, backend_name):
    """Safely get device_count for a torch backend."""
    backend = getattr(torch_module, backend_name, None)
    if backend is None:
        return 0
    device_count = getattr(backend, "device_count", None)
    if device_count is None:
        return 1 if _rf2ppi_backend_available(torch_module, backend_name) else 0
    try:
        return int(device_count())
    except Exception:
        return 0


def _select_rf2ppi_device():
    """Select an accelerator device only; never fall back to CPU.

    ZW810E/PPU vendor PyTorch is usually exposed through CUDA-compatible APIs,
    so CUDA is checked first. Intel torch-xpu may import successfully but return
    device_count == 0 on ZW810E/PPU, so XPU is used only when actually available.
    """
    if _rf2ppi_backend_available(torch, "cuda") and _rf2ppi_backend_device_count(torch, "cuda") > 0:
        return "cuda:0"
    if _rf2ppi_backend_available(torch, "xpu") and _rf2ppi_backend_device_count(torch, "xpu") > 0:
        return "xpu:0"
    if _rf2ppi_backend_available(torch, "npu") and _rf2ppi_backend_device_count(torch, "npu") > 0:
        return "npu:0"

    raise RuntimeError(
        "No accelerator device is available for RF2-PPI. "
        "CPU fallback is disabled. Check PPU runtime/library paths, device visibility, "
        "and run with /usr/local/bin/python outside conda."
    )


device = _select_rf2ppi_device()
model = RF2trackModule().to(device)

args = get_args()
num_seqs = args.number_seqs
chk_fn = args.model_file
input_fn = args.list_fn
map_location={'cuda:%d'%0: device}
checkpoint = torch.load(chk_fn, map_location=map_location, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)

results = {}
fp = open(input_fn, 'r')
lp = open(input_fn + '.log','w')
lp.write('Input_MSA\tInteraction_probability\tCompute_time\n')
model.eval()
table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
for line in fp:
    words = line.split()
    if len(words) != 2:
        lp.write(f'{words[0]} error! It does not have two columns\n')
        continue
    pair = words[0]

    msa = []
    seq_line = []
    with open(pair) as f_ent:
        for raw_line in f_ent:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            if raw_line[0] == '>':
                if seq_line:
                    msa.append(''.join(seq_line).translate(table))
                    seq_line = []
                    if len(msa) >= num_seqs:
                        break
                continue
            seq_line.append(raw_line)
        if seq_line and len(msa) < num_seqs:
            msa.append(''.join(seq_line).translate(table))

    t_length = len(msa[0])
    L1 = int(words[1])
    L2 = t_length - int(words[1])
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    alphabet = np.array(list('ARNDCQEGHILKMFPSTWYVX-'), dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i
    msa[msa > 21] = 21

    startT = time.time()
    network_input = prep_input(msa, L1, L2, device)
    with torch.no_grad():
        output_i = (None, None)
        for i_cycle in range(N_cycle):
            if i_cycle < N_cycle-1:
                return_raw=True
            else:
                return_raw=False
            input_i = _get_model_input(network_input, output_i, return_raw=return_raw, use_checkpoint=False)
            output_i = model(**input_i)
            if i_cycle < N_cycle - 1:
                continue

            logit_dist = output_i[0][0]
            prob_dist = torch.nn.Softmax(dim=1)(logit_dist)
            p_bind = prob_dist[:,:20].sum(dim=1) # (B, L, L)
            p_bind = p_bind * (1.0 - network_input['same_chain'].float()) # (B, L, L)
            p_bind = torch.nn.MaxPool2d(p_bind.shape[1:])(p_bind).view(-1)
            p_bind = torch.clamp(p_bind, min=0.0, max=1.0)
            _, N_seq, N_res = network_input['msa_orig'].shape
            prob = np.sum(prob_dist[0].permute(1,2,0).detach().cpu().numpy()[:L1,L1:,:20], axis=-1).astype(np.float16)
            
            results[pair] = prob
            endT = time.time()
            lp.write(pair + '\t' + str(round(p_bind.item(), 6)) + '\t' + str(round(endT-startT, 6)) + '\n')
np.savez_compressed(input_fn + '.npz', **results)
lp.close()
fp.close()
print(input_fn, 'done')
