python train_unsupervised.py \
    -c ./configs/SupCon_2_nn.yaml \
    --ckpt_dir ../checkpoints \
    --log_dir ../logs \
    --data_dir ../CATCH \
    --mem_dir ../CATCH/TRAIN_VAL_SET \
    --val_dir ../CATCH/VAL_SET