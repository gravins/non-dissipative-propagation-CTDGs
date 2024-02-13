#!/bin/bash
conda create -n ctan python=3.9
conda activate ctan

conda install gpustat -c conda-forge

#torch 2.0.1
python3 -m pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# torch geometric
python3 -m pip install  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
python3 -m pip install  torch_geometric==2.3.1 -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

python3 -m pip install ray==2.3.0
python3 -m pip install scikit-learn==1.2.2
python3 -m pip install pandas==1.5.3
python3 -m pip install tqdm==4.65.0
python3 -m pip install wandb==0.15.0
python3 -m pip install matplotlib==3.8.1

