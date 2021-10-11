from torch.utils.data import RandomSampler, DataLoader
import torch

class CoreDataModule():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):

        sampler = RandomSampler(
            self.train_dataset,
            replacement=True,
            num_samples=self.cfg.train_samples,
            generator=torch.Generator()
        )

        return DataLoader(
            self.train_dataset, batch_size=self.cfg.bs,
            num_workers=self.cfg.n_workers, drop_last=True, pin_memory=True,
            sampler=sampler
        )

    def val_dataloader(self):

        sampler = RandomSampler(
            self.val_dataset,
            replacement=True,
            num_samples=self.cfg.val_samples,
            generator=torch.Generator()
        )

        return DataLoader(
            self.val_dataset, batch_size=self.cfg.bs,
            num_workers=self.cfg.n_workers, drop_last=True, pin_memory=True,
            sampler=sampler
        )