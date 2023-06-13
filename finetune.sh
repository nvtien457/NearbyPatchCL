python finetune_isic.py \
    -c ./configs/ISIC.yaml \
    --ckpt_dir ../checkpoints \
    --log_dir ../logs \
    --data_dir ../CATCH \
    --mem_dir ../CATCH/TRAIN_VAL_SET \
    --val_dir ../ISIC_2019