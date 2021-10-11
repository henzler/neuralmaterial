import torch
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class DummyLogger:
    def update_metrics(*args):
        pass
    def log(*args):
        pass
    def log_dict(*args):
        pass
    def reset_metrics(*args):
        pass

class CoreLogger():
    def __init__(self):

        self.tb_dir = Path('tb')
        self.tb_dir.mkdir(exist_ok=True, parents=True)

        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self.metrics = {}

    def log(self, *args):
        pass

    def log_dict(self, outputs, mode, global_step, step):
        self.log_metrics(global_step, step, mode)
        self.log(outputs, mode, global_step)

    def update_metrics(self, metrics, mode):
        for k, v in metrics.items():
            v = v.detach().cpu().item()
            k = f'{mode}_{k}'

            if k in self.metrics:
                self.metrics[k] += v
            else:
                self.metrics[k] = v

    def reset_metrics(self):
        self.metrics = {}

    def log_metrics(self, epoch, step, mode):

        for k, v in self.metrics.items():
            if mode in k:
                self.writer.add_scalar(k, v / step, epoch)