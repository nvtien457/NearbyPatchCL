from tqdm import tqdm
import time

progress = tqdm(range(4, 10), desc='Training', initial=4, total=10)
for e in progress:
    print('Epoch', e)
    time.sleep(5)