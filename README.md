# ISL [ECCV-2022] 

[[Poster](https://drive.google.com/uc?id=1w0xS3TCQfTnuRsUQBPeK-bdX_-anw1iB)][[YouTube](https://www.youtube.com/watch?v=QfUJ9YTxyc0)][[Project Page](https://2022eccvisl.github.io/)][[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860354.pdf)][[Supplementary](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860354-supp.pdf)][[Springer Version](https://link.springer.com/chapter/10.1007/978-3-031-19809-0_21)]

This is the official repository that contains the source code for the paper "Balancing between Forgetting and Acquisition in Incremental Subpopulation Learning" (ECCV-2022). This code is refactored after the ECCV-2022 acceptance and re-tested on PyTorch 1.7.0 with 4 RTX3090 GPUs.

# News

[2022.09] The video for ECCV-2022 Presentation is posted!

[2022.07] Initial submission for the ECCV-2022 Camera Ready. We provide the complete reproduction procedure to let the potential researchers to replicate our proposed two-stage method for ISL.

# Introduction
The subpopulation shifting challenge, known as some subpopulations of a category that are not seen during training, severely limits the classification performance of the state-of-the-art convolutional neural networks. Thus, to mitigate this practical issue, we explore **incremental subpopulation learning (ISL)** to adapt the original model via incrementally learning the unseen subpopulations without retaining the seen population data. However, striking a great balance between subpopulation learning and seen population forgetting is the main challenge in ISL but is not well studied by existing approaches. These incremental learners simply use a pre-defined and fixed hyperparameter to balance the learning objective and forgetting regularization, but their learning is usually biased towards either side in the long run. In this paper, we propose a novel two-stage learning scheme to explicitly disentangle the acquisition and forgetting for achieving a better balance between subpopulation learning and seen population forgetting: in the first ''gain-acquisition'' stage, we progressively learn a new classifier based on the margin-enforce loss, which enforces the hard samples and population to have a larger weight for classifier updating and avoid uniformly updating all the population; in the second ''counter-forgetting'' stage, we search for the proper combination of the new and old classifiers by optimizing a novel objective based on proxies of forgetting and acquisition. We benchmark the representative and state-of-the-art non-exemplar-based incremental learning methods on a large-scale subpopulation shifting dataset for the first time. Under almost all the challenging ISL protocols, we significantly outperform other methods by a large margin, demonstrating our superiority to alleviate the subpopulation shifting problem.

We hope our proposed method can serve as an initial and novel baseline tailored to ISL for inspiring more future research in ISL.

![Illustrating of the two-stage training scheme](/images/structure_all_new_new.png)


# Requirements
This repository uses the following libraries:

- Python (3.6)
- Pytorch (1.7.0)
- torchvision (0.8.0)
- robustness (1.2.1.post2)
- networkx
- tqdm

We recommend to use the `conda` to create an specific environment for this project.

You first need to read the installation guideline from [BREEDS-Benchmarks](https://github.com/MadryLab/BREEDS-Benchmarks) and you should also be aware of the requirement in BREEDS-Benchmarks repo since our code is directly based on the [BREEDS-Benchmarks](https://github.com/MadryLab/BREEDS-Benchmarks) to generate the ISL experimental protocols.

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


# Datasets and Experimental Protocol Design

We leverage the latest [BREEDS dataset](https://openreview.net/forum?id=mQPBmvyAuk) in our experiments. BREEDS simulates the real-world subpopulation shifting based on the [ImageNet](http://www.image-net.org/), and it comprises **four different datasets: Entity-13, Entity-30, Living-17, and Non-Living-26**, with a total of 0.86 million (M) of images. However, BREEDS is not proposed for incremental subpopulation learning (ISL), so we need to further create the ISL-specific benchmark based on it. Since we focus on the incremental learner's performance in the sufficiently long run, hence in present work, our main testbeds are based on **Entity-13 and Entity-30** from BREEDS as they have the most number of subclasses, i.e., totally 260 and 240 subclasses respectively, and more than 0.6M images. To the best of our knowledge, this is the first time to leverage such large-scale datasets to investigate the ISL. 

<details>
<summary> Experimental Protocols Design</summary>

Entity-30 and Entity-13 have 30 and 13 classes where each class has 8 and 20 subclasses respectively. We design 3 protocols for each dataset. In the *base step*, the training set of each class comprises data from 4 and 10 subclasses for Entity-30 and Entity-13 respectively, the same as [breeds-benchmarks](https://openreview.net/forum?id=mQPBmvyAuk) to simulate subpopulation shifting. Then we split the rest of 120 and 130 unseen subclasses in each dataset respectively to create different protocols. For Entity-30, we design protocols with 4, 8, 15 incremental steps: in each step, for 4 Steps setup, each class is introduced with 1 unseen subclass; for 8 and 15 Steps setups, we randomly choose 15 and 8 out of 30 classes respectively to introduce with 1 unseen subclass. For Entity-13, we design protocols with 5, 10, 13 incremental steps: in each step, for 5 and 10 Steps setups, we introduce 2 and 1 unseen subclasses for each class respectively; For 13 Steps setup, we randomly sample 10 out of 13 classes to introduce with 1 unseen subclass. These designs simulate two scenarios: (1) all the classes are updated with at least 1 unseen subclass; (2) only a part of classes are updated with unseen subclasses. We denote the former as **even update** and the latter as **uneven update**. 

</details>

<details>
<summary> Dataset Generation for Different Incremental Steps</summary>

For the *base step* dataset genration, we exactly use the source part of each dataset in BREEDS, which is splited by the `split='rand'` in the [BREEDS-Benchmarks](https://github.com/MadryLab/BREEDS-Benchmarks). By doing so, our ISL exploration will be comparable to the existing BREEDS benchmark to see whether the incremental learning may help mitigate the subpopulation shifting problem. In BREEDS paper, they train on the source part of each dataset and then test on the target part (with unseen subpopulations) to demonstrate the subpopulation shifting problem, where the latter's performance drops mostly larger than 30%. In our paper, since we want to explore whether we can mitigate the subpopulation shifting by incremental learning, hence we split the target part of each dataset and adapt our original model on them in an incremental learning manner. We want to investigate whether these unseen subpopulaitons' performance can be improved while the seen population's performance can be still maintained without catastrophic forgetting.

</details>

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

# Pretrained Model of the *base step*
We follow the same training receipt detailed in [BREEDS]((https://openreview.net/forum?id=mQPBmvyAuk)) supplementary to train the *base step model* and  we use the same *base step model* to perform incremental subpopulation learning for all the compared methods. We first replicate the results in [BREEDS]((https://openreview.net/forum?id=mQPBmvyAuk)) by exactly following their training receipt on the source split of every dataset, i.e., Entity13 and Entity30, and we do observe the critical subpopulation shifting problem as demonstrated in [BREEDS]((https://openreview.net/forum?id=mQPBmvyAuk)). 

Since [BREEDS]((https://openreview.net/forum?id=mQPBmvyAuk)) do not provide any pretrained models on the source part of each dataset for their paper, thus to faciliate the research, we provide our pretrained *base step model* here: 

Entity30: [[google drive](https://drive.google.com/file/d/1O6NFbqK55m3LP697TIjjjotUl_jHOn0c/view?usp=sharing)] 

Entity13: [[google drive](https://drive.google.com/file/d/1jlJ2XDxt4U_itLiL09mCaIL1bEpTA_N_/view?usp=sharing)]

You can download them and put them in a folder, e.g., `ckpts/entity30/` and `ckpts/entity13/`. You will pass their paths by `--base_step_pretrained_path` when you are running the experiments later.

If you want to train these *base step* models from scratch, you can refer to `train_imagenet_vanilla_breeds_dataset_standard_data_augmentation_300_epoch.py` for more details.

# Reproduction Guideline
To replicate the results of our proposed two-stage method on a specific ISL protocol, e.g., the 15 Steps Entity30, run the following command:

```
python main_isl_15tasks_entity30.py \
    --ds_name entity30 \
    --inc_step_num 15 \
    --exp_name any_name_you_want \
    --info_dir /path/to/your/BREEDS-Benchmarks/imagenet_class_hierarchy/modified \
    --data_dir /path/to/your/ILSVRC2012_Data \
    --base_step_pretrained_path /ckpts/entity30/model_best.pth.tar
```

The command is almost the same for other ISL protocols and you only need to change the name of the python file, `--ds_name` and `--inc_step_num` accordingly. If you change to perform the experiments on entity13, do remember to change the `--base_step_pretrained_path` to `/ckpts/entity13/model_best.pth.tar`. The training hyperparameters (learning rate, weight decay, momentum) chosen by the Continual Hyperparameter Framework (CHF) for our proposed method is already set as the default value in each python file. For more information, discussions and experimental details, please refer to our supplementary. 


# Cite us
If you use this repository and find it useful for your research, please consider to cite:
```
@InProceedings{10.1007/978-3-031-19809-0_21,
author="Liang, Mingfu and Zhou, Jiahuan and Wei, Wei and Wu, Ying",
editor="Avidan, Shai and Brostow, Gabriel and Ciss{\'e}, Moustapha and Farinella, Giovanni Maria and Hassner, Tal",
title="Balancing Between Forgetting and Acquisition in Incremental Subpopulation Learning",
booktitle="Computer Vision -- ECCV 2022",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="364--380",
isbn="978-3-031-19809-0"
}
```

# Acknowledgements

We want to thanks the author of [BREEDS dataset](https://openreview.net/forum?id=mQPBmvyAuk) for providing such a large scale dataset to pave the way for our timely study of the subpopulation shifting problem.

Mingfu want to sincerely thank the author of [BREEDS dataset](https://openreview.net/forum?id=mQPBmvyAuk) for timely response regarding the replication of their results on Oct. 2021, which largely speed up the progress of this research project. 

The code in this project is largerly developed upon [BREEDS-Benchmarks](https://github.com/MadryLab/BREEDS-Benchmarks). The code for reproducing the result of the BREEDS is based on the [DIANet](https://github.com/gbup-group/DIANet) and [IEBN](https://github.com/gbup-group/IEBN) and [pytorch-classification](https://github.com/bearpaw/pytorch-classification).



