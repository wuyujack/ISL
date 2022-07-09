# ISL

This is the repository that contains the source code for the paper "Balancing between Forgetting and Acquisition in Incremental Subpopulation Learning" (ECCV-2022). This code is refactored after the ECCV-2022 acceptance.


# Requirements
This repository uses the following libraries:

- Python (3.6)
- Pytorch (1.7.0)
- torchvision (0.8.0)
- robustness (1.2.1.post2)
- networkx
- tqdm

We recommend first read the installation guideline from https://github.com/MadryLab/BREEDS-Benchmarks and also be aware of the requirement in breeds-benchmarks since our code is directly based on the breeds-benchmarks to generate the ISL experimental protocols.

1. git clone the `git clone https://github.com/MadryLab/BREEDS-Benchmarks.git` to get the BREEDS benchmark's statistics.

# Dataset
We use the same dataset as in breeds-benchmarks, i.e., ILSVRC2012 dataset, to generate the BREEDS datasets following the breeds-benchmarks. Please Download the [ImageNet](http://www.image-net.org/) dataset from the official website.

# Pretrained Model of the base step
We follow the same training recipt in BREEDS to train the base step model and use the same base step model to perform incremental learning. We first replicate the results in BREEDS and to faciliate the research, we provide our pretrained model here:

# Training Guideline

# Testing Guideline

# Cite us
If you use this repository, please consider to cite
