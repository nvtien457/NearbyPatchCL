import os

for f in os.listdir('../CATCH/FINETUNE/VAL_SET/Dermis'):
    # if f == 'Plasmacytoma_31_1_99551.0_22998.0.jpg':
    #     print('yes')
    if f.startswith('Plasmacytoma'):
        print(f)

print('no')