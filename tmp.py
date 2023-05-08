import os

train_dir = '../CATCH/TRAIN_SET'
data_dir  = './data/full'

for cancer in os.listdir(train_dir):
    with open(f'{data_dir}/{cancer}_full.txt', 'w') as f:
        for img in os.listdir(os.path.join(train_dir, cancer)):
            name = img.replace('.jpg', '')
            f.write(f'TRAIN_SET/{cancer}/{name}\n')