# import torch
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
from tensorboardX import SummaryWriter
from tensorflow.python.summary.summary_iterator import summary_iterator

from torch import Tensor
from collections import OrderedDict
import os
import numpy as np
from .plotter import Plotter


class Logger(object):
    def __init__(self, log_dir, tensorboard=True, matplotlib=True):

        self.reset(log_dir, tensorboard, matplotlib)

    def reset(self, log_dir=None, tensorboard=True, matplotlib=True):

        if log_dir is not None: self.log_dir=log_dir 
        self.writer = SummaryWriter(log_dir=self.log_dir) if tensorboard else None
        self.plotter = Plotter() if matplotlib else None
        self.counter = OrderedDict()
        self.data_dict = dict()

    def update_scalers(self, ordered_dict):

        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                ordered_dict[key] = value.item()

            if self.counter.get(key) is None:
                self.counter[key] = 1
            else:
                self.counter[key] += 1

            if self.writer:
                self.writer.add_scalar(key, value, self.counter[key])

        if self.plotter: 
            self.plotter.update(ordered_dict)
            self.plotter.save(os.path.join(self.log_dir, 'plotter.svg'))

    def load_event(self, event:str, epoch:int):
        '''
        Load history log
        Parameter:
            + event: Path of event file
            + epoch: The last epoch want to load
        '''
        for summary in summary_iterator(event):
            for v in summary.summary.value:
                if self.data_dict.get(v.tag) is None:
                  self.data_dict[v.tag] = [v.simple_value]
                else:
                  self.data_dict[v.tag].append(v.simple_value)

        if epoch:   epoch = epoch
        else:       epoch = self.data_dict['epoch'][-1]

        num_epochs = len(self.data_dict['epoch'])
        if epoch < 0 or epoch > self.data_dict['epoch'][-1]:
          raise Exception('Epoch {} out of range [{}, {}]'.format(epoch, 0, self.data_dict['epoch'][-1]))

        for k, v in self.data_dict.items():
          iters_per_epoch = len(v) // num_epochs
          self.data_dict[k] = v[:((epoch+1)*iters_per_epoch)]

        if self.plotter: 
          for k, v in self.data_dict.items():
            for i in v:
              if self.counter.get(k) is None:
                  self.counter[k] = 1
              else:
                  self.counter[k] += 1

              if self.writer:
                  self.writer.add_scalar(k, i, self.counter[k])
                  
              self.plotter.update({k: i})
          self.plotter.save(os.path.join(self.log_dir, 'plotter.svg'))

        avg_loss = np.average(np.array(self.data_dict['loss']).reshape(-1, len(self.data_dict['loss']) // num_epochs), axis=1).tolist()

        return min(avg_loss)