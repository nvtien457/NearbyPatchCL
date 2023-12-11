### Openslide Installation 
Tutorial: https://www.youtube.com/watch?v=QntLBvUZR5c/


### RAW SVS
Raw SVS files structure:
```
├── RAW_TRAIN_SVS
|   ├── Histiocytoma_03_1.svs
|   ├── Histiocytoma_07_1.svs
|   ├── ...
|   ├── Trichoblastoma_36_1.svs
```

### Create Image patches for training self-supervised
```
!python create_train_patches.py
```
