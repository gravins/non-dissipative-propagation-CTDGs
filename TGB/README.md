<!-- # TGB -->
![TGB logo](imgs/logo.png)

**Temporal Graph Benchmark for Machine Learning on Temporal Graphs** (NeurIPS 2023 Datasets and Benchmarks Track)
<h4>
	<a href="https://arxiv.org/abs/2307.01026"><img src="https://img.shields.io/badge/arXiv-pdf-yellowgreen"></a>
	<a href="https://pypi.org/project/py-tgb/"><img src="https://img.shields.io/pypi/v/py-tgb.svg?color=brightgreen"></a>
	<a href="https://tgb.complexdatalab.com/"><img src="https://img.shields.io/badge/website-blue"></a>
	<a href="https://docs.tgb.complexdatalab.com/"><img src="https://img.shields.io/badge/docs-orange"></a>
</h4> 


Overview of the Temporal Graph Benchmark (TGB) pipeline:
- TGB includes large-scale and realistic datasets from five different domains with both dynamic link prediction and node property prediction tasks.
- TGB automatically downloads datasets and processes them into `numpy`, `PyTorch` and `PyG compatible TemporalData` formats. 
- Novel TG models can be easily evaluated on TGB datasets via reproducible and realistic evaluation protocols. 
- TGB provides public and online leaderboards to track recent developments in temporal graph learning domain.

![TGB dataloading and evaluation pipeline](imgs/pipeline.png)

**To submit to [TGB leaderboard](https://tgb.complexdatalab.com/), please fill in this [google form](https://forms.gle/SEsXvN1QHo9tSFwx9)**

**See all version differences and update notes [here](https://tgb.complexdatalab.com/docs/update/)**

### Annoucements

**Excited to annouce that TGB has been accepted to NeurIPS 2023 Datasets and Benchmarks Track!**

Thanks to everyone for your help in improving TGB! we will continue to improve TGB based on your feedback and suggestions. 


**Please update to version `0.9.0`**

#### version `0.9.0`

Added the large `tgbn-token` dataset with 72 million edges to the `nodeproppred` dataset. 

fixed errors in `tgbl-coin` and `tgbl-flight` where a small set of edges are not sorted chronologically. Please update your dataset version for them to version 2 (will be promted in terminal).


### Pip Install

You can install TGB via [pip](https://pypi.org/project/py-tgb/). **Requires python >= 3.9**
```
pip install py-tgb
```

### Links and Datasets

The project website can be found [here](https://tgb.complexdatalab.com/).

The API documentations can be found [here](https://shenyanghuang.github.io/TGB/).

all dataset download links can be found at [info.py](https://github.com/shenyangHuang/TGB/blob/main/tgb/utils/info.py)

TGB dataloader will also automatically download the dataset as well as the negative samples for the link property prediction datasets.

if website is unaccessible, please use [this link](https://tgb-website.pages.dev/) instead.


### Running Example Methods

- For the dynamic link property prediction task, see the [`examples/linkproppred`](https://github.com/shenyangHuang/TGB/tree/main/examples/linkproppred) folder for example scripts to run TGN, DyRep and EdgeBank on TGB datasets.
- For the dynamic node property prediction task, see the [`examples/nodeproppred`](https://github.com/shenyangHuang/TGB/tree/main/examples/nodeproppred) folder for example scripts to run TGN, DyRep and EdgeBank on TGB datasets.
- For all other baselines, please see the [TGB_Baselines](https://github.com/fpour/TGB_Baselines) repo.

### Acknowledgments
We thank the [OGB](https://ogb.stanford.edu/) team for their support throughout this project and sharing their website code for the construction of [TGB website](https://tgb.complexdatalab.com/).


### Citation

If code or data from this repo is useful for your project, please consider citing our paper:
```
@article{huang2023temporal,
  title={Temporal graph benchmark for machine learning on temporal graphs},
  author={Huang, Shenyang and Poursafaei, Farimah and Danovitch, Jacob and Fey, Matthias and Hu, Weihua and Rossi, Emanuele and Leskovec, Jure and Bronstein, Michael and Rabusseau, Guillaume and Rabbany, Reihaneh},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
<!-- 

### Install dependency
Our implementation works with python >= 3.9 and can be installed as follows

1. set up virtual environment (conda should work as well)
```
python -m venv ~/tgb_env/
source ~/tgb_env/bin/activate
```

2. install external packages
```
pip install pandas==1.5.3
pip install matplotlib==3.7.1
pip install clint==0.5.1
```

install Pytorch and PyG dependencies (needed to run the examples)
```
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

3. install local dependencies under root directory `/TGB`
```
pip install -e .
```


### Instruction for tracking new documentation and running mkdocs locally

1. first run the mkdocs server locally in your terminal 
```
mkdocs serve
```

2. go to the local hosted web address similar to
```
[14:18:13] Browser connected: http://127.0.0.1:8000/
```

Example: to track documentation of a new hi.py file in tgb/edgeregression/hi.py


3. create docs/api/tgb.hi.md and add the following
```
# `tgb.edgeregression`

::: tgb.edgeregression.hi
```

4. edit mkdocs.yml 
```
nav:
  - Overview: index.md
  - About: about.md
  - API:
	other *.md files 
	- tgb.edgeregression: api/tgb.hi.md
```

### Creating new branch ###
```
git fetch origin

git checkout -b test origin/test
```

### dependencies for mkdocs (documentation)
```
pip install mkdocs
pip install mkdocs-material
pip install mkdocstrings-python
pip install mkdocs-jupyter
pip install notebook
```


### full dependency list
Our implementation works with python >= 3.9 and has the following dependencies
```
pytorch == 2.0.0
torch-geometric == 2.3.0
torch-scatter==2.1.1
torch-sparse==0.6.17
torch-spline-conv==1.2.2
pandas==1.5.3
clint==0.5.1
``` -->
