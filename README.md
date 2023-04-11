# Contrastive-Learning

Người thực hiện: Nguyễn Văn Tiến

Implement các thuật toán contrastive & non-contrastive learning: SimCLR, Moco, SimSiam, BYOL, SimTriplet.

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

- Dùng hàm get_aug trong thư mục **`augmentations`** để khởi tạo TwoCropsTransform(transform) (2 transforms cùng 1 lúc). Hiện tại đã implement được Moco, SimCLR, BYOL.

- Thực hiện chỉnh sửa các tên, biến cần truyền vào của (model, augmentation, loss, dataset, optimizer, train) ở trong các file .yaml trong thư mục **`configs`**.

- Thư mục **`data`** chứa các dataset folder, trong mỗi folder chứa các file .txt (đường dẫn tới ảnh).

- Dùng hàm get_criterion trong thư mục **`losses`** để khởi tạo Criterion. Ngoài ra còn implement các hàm loss không có sẵn trên Pytorch.

- Trong thư mục **`models`**, dùng hàm get_model để khởi tạo model, get_backbone để khởi tạo backbone (nếu muốn giữ y nguyên backbone gốc thì castrate = False trong config).

- Trong thư mục **`optimizers`**, dùng hàm get_optimizer để khởi tạo optimizer, get_scheduler để khởi tạo scheduler. Hiện mới có LRScheduler được implement, warmup_epoch có thể bằng 0.

- Thư mục **`tools`** chứa đủ thứ: accuracy, argument, logger, meter, ...

- Thư mục **`trainer`** dùng để xác định cách mỗi model sẽ lấy data từ dataloader, forward, trả về những gì, tính accuracy sao, ... Trả về đều dưới dạng dict, luôn có 1 key loss.

# 1. Train unsupervised setting

- Sử dụng file train_unsupervised.py. 
- XÁC ĐỊNH các đường dẫn tới logs, checkpoints, data folder trước khi train.
- Kiểm tra lại file .yaml (trong folder configs) được dùng.

```
!python train_unsupervised.py \
    -c ./configs/SimCLR_unsup.yaml \
    --data_dir /content \
    --ckpt_dir ../NVT_checkpoints \
    --log_dir ../NVT_logs
```