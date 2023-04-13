python train_unsupervised.py \
    -c ./configs/SimSiam_unsup.yaml \
    --ckpt_dir ../checkpoints \
    --log_dir ../logs \
    --data_dir ../CATCH \
    --mem_dir ../CATCH/TRAIN_VAL_SET \
    --val_dir ../CATCH/VAL_SET \
    --hide_progress \
    --debug --debug_subset_size 8