# ISL [ECCV-2022]

This is the official repository that contains the source code for the paper "Balancing between Forgetting and Acquisition in Incremental Subpopulation Learning" (ECCV-2022). This code is refactored after the ECCV-2022 acceptance and re-test on PyTorch 1.7.0.

# Introduction
The subpopulation shifting challenge, known as some subpopulations of a category that are not seen during training, severely limits the classification performance of the state-of-the-art convolutional neural networks. Thus, to mitigate this practical issue, we explore incremental subpopulation learning~(ISL) to adapt the original model via incrementally learning the unseen subpopulations without retaining the seen population data. However, striking a great balance between subpopulation learning and seen population forgetting is the main challenge in ISL but is not well studied by existing approaches. These incremental learners simply use a pre-defined and fixed hyperparameter to balance the learning objective and forgetting regularization, but their learning is usually biased towards either side in the long run. In this paper, we propose a novel two-stage learning scheme to explicitly disentangle the acquisition and forgetting for achieving a better balance between subpopulation learning and seen population forgetting: in the first ''gain-acquisition'' stage, we progressively learn a new classifier based on the margin-enforce loss, which enforces the hard samples and population to have a larger weight for classifier updating and avoid uniformly updating all the population; in the second ''counter-forgetting'' stage, we search for the proper combination of the new and old classifiers by optimizing a novel objective based on proxies of forgetting and acquisition. We benchmark the representative and state-of-the-art non-exemplar-based incremental learning methods on a large-scale subpopulation shifting dataset for the first time. Under almost all the challenging ISL protocols, we significantly outperform other methods by a large margin, demonstrating our superiority to alleviate the subpopulation shifting problem.

![Illustrating of the two-stage training scheme](/images/structure_all_new_new.png)


# Requirements
This repository uses the following libraries:

- Python (3.6)
- Pytorch (1.7.0)
- torchvision (0.8.0)
- robustness (1.2.1.post2)
- networkx
- tqdm

We recemmend to use the conda to create an specific environment for this project.

You first need to read the installation guideline from [BREEDS-Benchmarks](https://github.com/MadryLab/BREEDS-Benchmarks) and You should also be aware of the requirement in BREEDS-Benchmarks repo since our code is directly based on the BREEDS-Benchmarks to generate the ISL experimental protocols.

0. Clone this repo: `https://github.com/wuyujack/ISL.git`
1. You can git clone the BREEDS-Benchmarks anywhere in your computer, just use: `git clone https://github.com/MadryLab/BREEDS-Benchmarks.git` to get the BREEDS benchmark's splits information from their official GitHub repo. The path of your BREEDS-Benchmarks folder will be passed by the `--info_dir` when you are running the experiments later. 
2. Install dependencies:
    ```
    conda create -n ISL python=3.6
    conda activate ISL
    conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
    pip install robustness
    conda install pandas
    conda install matplotlib
    pip install networkx
    ```


# Data Preparation
We use the same dataset as in [breeds-benchmarks](https://openreview.net/forum?id=mQPBmvyAuk), i.e., ILSVRC2012 dataset, to generate the BREEDS datasets following the breeds-benchmarks. Please Download the [ImageNet](http://www.image-net.org/) dataset from the official website. The dataset should be organized like the following:

```
`-- ISLVRC2012_Data
    |-- train
    |   |-- n03388549
    |   |-- n07615774
    |   |-- n02113624
    |   |-- ... 
    `-- val
        |-- n03388549
        |-- n07615774
        |-- n02113624
        |-- ... 
```

The path of your ILSVRC2012 dataset will be passed by the `--data_dir` when you are running the experiments later.

# Pretrained Model of the base step
We follow the same training recipt in BREEDS to train the base step model and use the same base step model to perform incremental learning. We first replicate the results in [BREEDS]((https://openreview.net/forum?id=mQPBmvyAuk)) by exactly following their training recipt on the source split of every dataset, i.e., Entity13 and Entity30. To faciliate the research, we provide our pretrained base step model here: 

Entity30: [[google drive](https://drive.google.com/file/d/1O6NFbqK55m3LP697TIjjjotUl_jHOn0c/view?usp=sharing)] 

Entity13: [[google drive](https://drive.google.com/file/d/1jlJ2XDxt4U_itLiL09mCaIL1bEpTA_N_/view?usp=sharing)]


# Reproduction Guideline
To replicate the results of our proposed two-stage method on a specific experimental protocol, e.g., the 15 Steps Entity30, run the following command:

```
python main_isl_15tasks_entity30.py \
    --ds_name entity30 \
    --inc_step_num 15 \
    --exp_name any_name_you_want \
    --info_dir /path/to/your/BREEDS-Benchmarks/imagenet_class_hierarchy/modified \
    --data_dir /path/to/your/ILSVRC2012_Data \
    --base_step_pretrained_path /ckpts/entity30/model_best.pth.tar
```

The command is the same for other experimental protocols and you only need to change the corresponding `--ds_name` and `--inc_step_num`. If you change to perform the experiments on entity13, do remember to change the `--base_step_pretrained_path` to `/ckpts/entity13/model_best.pth.tar`. The hyperparameters chosen by the Continual Hyperparameter Framework (CHF) for our proposed method is already set as the default value in each file. For more information and details, please refer to our supplementary. 


# Cite us
If you use this repository, please consider to cite
