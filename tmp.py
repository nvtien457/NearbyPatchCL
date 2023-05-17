import os
from tensorflow.python.summary.summary_iterator import summary_iterator

event = '../logs/completed_Tien_SupCon0_20_256_0508033246/events.out.tfevents.1683516766.selab2'

data_dict = {}
for summary in summary_iterator(event):
    for v in summary.summary.value:
        if data_dict.get(v.tag) is None:
            data_dict[v.tag] = [v.simple_value]
        else:
            data_dict[v.tag].append(v.simple_value)

num_epochs = len(data_dict['epoch'])
iter_per_epochs = len(data_dict['lr']) // num_epochs

for e in range(70, 80):
    neg = True
    for i, l in enumerate(data_dict['loss'][e*iter_per_epochs:(e+1)*iter_per_epochs]):
        if l > -5:
            print(e)
            neg = False
            break

    if neg == False:
        break
