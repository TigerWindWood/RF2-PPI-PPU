## Component Versions

RF2-PPI-PPU was tested on the Zhenwu 810E PPU platform with the following software stack:

```text
inference-xpu-pytorch:26.01
torch2.9-cu129
python 3.12
```

For compatibility with the original RF2-PPI dependency stack, we recommend creating a separate conda environment with Python 3.9.

## Installation

```bash
# 1. Create and activate the environment
conda create -n rf2ppi-ppu python=3.9
conda activate rf2ppi-ppu

# 2. Install HH-suite
conda install -c conda-forge -c bioconda hhsuite

# 3. Install Python dependencies
pip install biopython==1.83 -i https://pypi.org/simple
pip install numpy==1.26.4 pandas==2.2.2 scipy==1.11.4 scikit-learn==1.3.2 tqdm==4.64.1 h5py==3.10.0 networkx==2.8.8 einops==0.6.1

# 4. Clone the repository
git clone https://github.com/TigerWindWood/RF2-PPI-PPU.git
cd RF2-PPI-PPU

# 5. Download the RF2-PPI model weights
cd src/models
wget --no-check-certificate https://conglab.swmed.edu/humanPPI/downloads/RF2-PPI.pt
cd ../..
```

After installation, the model checkpoint should be located at:

```text
RF2-PPI-PPU/src/models/RF2-PPI.pt
```

## Usage

```bash
# Activate the environment
conda activate rf2ppi-ppu

# Run RF2-PPI-PPU inference
python [/path/to/]RF2-PPI-PPU/src/predict_list_PPI.py \
    -list_fn [input_file] \
    -model_file [/path/to/]RF2-PPI-PPU/src/models/RF2-PPI.pt
```

Example:

```bash
python /data/RF2-PPI-PPU/src/predict_list_PPI.py \
    -list_fn test_pairs_input \
    -model_file /data/RF2-PPI-PPU/src/models/RF2-PPI.pt
```

The input file should contain two columns: the paired MSA file and the length of the first protein.

```text
protein_paired_msas/P50579_Q9Y490.i90.a3m    512
protein_paired_msas/B2RTY4_Q9Y4C1.i90.a3m    430
protein_paired_msas/Q15746_Q8WYB5.i90.a3m    621
```

For an input file named `test_pairs_input`, the output files will be:

```text
test_pairs_input.npz
test_pairs_input.log
test_pairs_input.failed
```

The `.log` file contains the RF2-PPI prediction results:

```text
Input_MSA    Interaction_probability    Compute_time
```
