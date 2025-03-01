# <ins>C</ins>ontinuous-<ins>T</ins>ime Graph <ins>A</ins>nti-Symmetric <ins>N</ins>etwork (CTAN).
Official reference implementation of our paper [___"Long Range Propagation on Continuous-Time Dynamic Graphs"___](https://proceedings.mlr.press/v235/gravina24a.html) accepted at ICML 2024,
and [___"Effective Non-Dissipative Propagation for Continuous-Time Dynamic Graphs"___](https://openreview.net/forum?id=zAHFC2LNEe) accepted at the Temporal Graph Learning Workshop @ NeurIPS 2023

Please consider citing us
<!---
	  @inproceedings{gravina2024ctan,
	     title={Long Range Propagation on Continuous-Time Dynamic Graphs},
	     author={Alessio Gravina and Giulio Lovisotto and Claudio Gallicchio and Davide Bacciu and Claas Grohnfeldt},
	     booktitle={Forty-first International Conference on Machine Learning},
	     year={2024},
	     url={https://openreview.net/forum?id=gVg8V9isul}
	  }
-->
	  @InProceedings{gravina2024ctan,
	    title = {Long Range Propagation on Continuous-Time Dynamic Graphs},
	    author = {Gravina, Alessio and Lovisotto, Giulio and Gallicchio, Claudio and Bacciu, Davide and Grohnfeldt, Claas},
	    booktitle = {Proceedings of the 41st International Conference on Machine Learning},
	    pages = {16206--16225},
	    year = {2024},
	    editor = {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
	    volume = {235},
	    series = {Proceedings of Machine Learning Research},
	    month = {21--27 Jul},
	    publisher = {PMLR},
	    pdf = {https://raw.githubusercontent.com/mlresearch/v235/main/assets/gravina24a/gravina24a.pdf},
	    url = {https://proceedings.mlr.press/v235/gravina24a.html},
	  }
### Credits
Many thanks to **Giulio Lovisotto** ([Github](https://github.com/giuliolovisotto) / [Homepage]( https://giuliolovisotto.github.io/)) for his invaluable help on this project.


## Requirements
_Note: we assume Miniconda/Anaconda is installed, otherwise see this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for correct installation. The proper Python version is installed during the first step of the following procedure._

There are two ways to install the required packages:
1. Run the script ```create_env.sh```

or

2. Run ``` conda env create -f env.yml ``` followed by  ``` conda activate ctan ```

Then

   ``` cd TGB; python3 -m pip install . ```

## How to reproduce our results
- Sequence Classification on Temporal Path Graph

    ``` python3 run_sequence.py ```

- Classification on Temporal Pascal-VOC

    ``` python3 run_pascal.py ```

- Future link prediction tasks (ie, Wikipedia, Reddit, LsatFM, MOOC)

    ``` python3 run_link_prediction.py ```

- Temporal Graph Benchmark

    ``` python3 run_tgb.py ```


### Usage
This repo builds on [CTDG-learning-framework](https://github.com/gravins/CTDG-learning-framework), a framework to easlity experiment with Graph Neural Networks (GNNs) in the temporal domain. Please refer to it for more details.

## Repository structure
The repository is structured as follows:

    ├── README.md                <- The top-level README.
    │
    ├── env.yml                  <- The conda environment requirements.
    ├── create_env.sh            <- The script for building the environment.
    │
    ├── main.py                  <- The main file for Sequence Classification on Temporal Path Graph, Classification on Temporal Pascal-VOC, and Future link prediction tasks.
    ├── main_tgb_ctan.py         <- The main file for the Temporal Graph Benchmark.
    ├── negative sampler.py      <- Implements the class of the negative sampler.
    ├── train_link.py            <- Implements the code responsible for training and evaluation of the models in link prediction tasks.
    ├── train_sequence.py        <- Implements the code responsible for training and evaluation of the models in sequence prediction tasks.
    ├── utils.py                 <- Contains the code for multiple utilities.
    │
    ├── run_link_prediction.py   <- The script for reproducing the Future link prediction results.
    ├── run_sequence.py          <- The script for reproducing the Sequence Classification on Temporal Path Graph.
    ├── run_pascal.py            <- The script for reproducing the Temporal Pascal-VOC results.
    ├── run_tgb.py               <- The script for reproducing the Temporal Graph Benchmark results.
    │
    ├── confs                    <- Contains the the evaluated model configurations.
    │      
    ├── data                     <- Contains the data for Sequence Classification on Temporal Path Graph and Classification on Temporal Pascal-VOC tasks.
    │      
    ├── datasets                 <- Contains the code to create and load the datasets.
    │      
    ├── models                   <- Contains the code to implement CTAN and other C-TDG GNNs.
    │      
    └── TGB                      <- Contains the code of the TGB utilities.



