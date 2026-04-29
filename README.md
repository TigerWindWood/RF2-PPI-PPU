# RF2-PPI-PPU
RF2-PPI-PPU: an optimized RoseTTAFold2-PPI inference and preprocessing pipeline adapted for Zhenwu 810E PPU platforms.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/CongLabCode/RoseTTAFold2-PPI.git
   ```

2. Download the weights to RoseTTAFold2-PPI/src/model:

   ```bash
   cd RoseTTAFold2-PPI/src/models
   wget --no-check-certificate https://conglab.swmed.edu/humanPPI/downloads/RF2-PPI.pt
   ```

3. Choose one of the following installation methods:

   A. Download our singularity image (hhsuite is needed if you want to use our paired MSA generation script):
      
      ```bash
      cd RoseTTAFold2-PPI/
      wget --no-check-certificate https://conglab.swmed.edu/humanPPI/downloads/SE3nv-20230612.sif
      conda install -c conda-forge -c bioconda hhsuite
      ```

   B. Install conda environment (if cannot use singularity):
      
      ```bash
      conda create -n rf2ppi python=3.9
      conda activate rf2ppi
      conda install -c conda-forge -c bioconda hhsuite
      pip install numpy==1.21.2
      pip install pandas==1.5.3
      pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
      pip install biopython==1.79
      pip install scipy==1.7.1
      pip install einops
      ```

<br>
