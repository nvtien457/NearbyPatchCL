# SELF-SUPERVISED LEARNING APPROACH FOR DIGITAL PATHOLOGY IMAGE ANALYSIS

Nguyễn Văn Tiến - 19125124 \
Lê Gia Bảo - 19125079 

Implement self-supervised learning methods: SimCLR, MoCo, SimSiam, BYOL, SimTriplet, BarlowTwins, SupCon, MICLe.

Dataset:
+ Train set: https://studenthcmusedu-my.sharepoint.com/:f:/g/personal/19125079_student_hcmus_edu_vn/El5OuDuOcZ9OsH7Og4n06B0Bxx2q5P3riwhB6Qhc_ezr1w?e=HBewsU

+ Finetune set: https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/19125079_student_hcmus_edu_vn/EYM3yOR5X69AhvlbYIW-FeIB1X0BOYhKWXHWjJg7ElI-hg?e=wvgjjo

+ Test set: https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/19125079_student_hcmus_edu_vn/Edn14ViXfSdGhw47RS5KXjkB3apJagBvU-Gn9TodKkq6yw?e=J8q2N8
+ Pretrain model : https://studenthcmusedu-my.sharepoint.com/:f:/g/personal/19125079_student_hcmus_edu_vn/Er_jhqx23DBFoKNO_e3OYSgBYcRjYAUOjHZLJ7IqPuvmOA?e=gSIAUb

## Cây thư mục

```bash
├── README.md
├── augmentations
│   ├── __init__.py
│   ├── byol_aug.py
│   ├── moco_aug.py
│   └── simclr_aug.py
|
├── configs
│   ├── BYOL_unsup.yaml
│   ├── MoCo_unsup.yaml
│   ├── SimCLR_unsup.yaml
│   ├── SimSiam_unsup.yaml
│   └── SimTriplet_unsup.yaml
|
├── data
|
├── datasets
│   ├── __init__.py
│   └── catch_dataset.py
|
├── losses
│   ├── NT_Xent.py
│   ├── __init__.py
│   └── neg_cosine.py
|
├── models
│   ├── __init__.py
│   ├── byol.py
│   ├── moco.py
│   ├── simclr.py
│   ├── simsiam.py
│   └── simtriplet.py
|
├── optimizers
│   ├── __init__.py
│   ├── larc.py
│   ├── lars.py
│   ├── lars_simclr.py
│   └── lr_scheduler.py
|
├── tools
│   ├── __init__.py
│   ├── accuracy.py
│   ├── arguments.py
│   ├── average_meters.py
│   ├── file_exist_fn.py
│   ├── knn_monitor.py
│   ├── logger.py
│   └── plotter.py
|
├── trainer
│   ├── __init__.py
│   ├── byol.py
│   ├── moco.py
│   ├── simclr.py
│   ├── simsiam.py
│   └── simtriplet.py
|
├── train_unsupervised.py
└── unsup.sh
```
# 1. Fully-supervised training setting

- Use file **sup.sh**. 
- Determine the path of logs, checkpoints, data folder.
- Config setting for training is saved in **.yaml** file.

```
!python train_supervised.py \
    -c ./configs/SimCLR_unsup.yaml \
    --data_dir /content \
    --ckpt_dir path \
    --log_dir path
```
# 1. Pre-train unsupervised setting

- Use file **unsup.sh**. 
- Determine the path of logs, checkpoints, data folder.
- Config setting for training is saved in **.yaml** file.

```
!python train_unsupervised.py \
    -c ./configs/SimCLR_unsup.yaml \
    --data_dir /content \
    --ckpt_dir path \
    --log_dir path
```

- To train method NearbyPatchCL(N=X), use config files **NEAR_X_D.yaml**

# 2. Finetune model

- Use file **finetune.sh** 
```
!python train_linear.py \
    -c ./configs/finetune.yaml \
    --data_dir Path 
```
# 3. Evaluate testing
- Use file **test.sh** 
```
!python test.py \
    -c ./configs/test.yaml \
    --test_dir Path
```
