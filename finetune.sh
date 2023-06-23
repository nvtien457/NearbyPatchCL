python train_linear.py \
    -c ./configs/SupCon0_finetune.yaml \
    --ckpt_dir ../checkpoints \
    --log_dir ../logs \
    --data_dir ../CATCH \
    --mem_dir ../CATCH/TRAIN_VAL_SET \
    --val_dir ../CATCH/FINETUNE