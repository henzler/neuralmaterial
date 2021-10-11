from lib.dataset import MaterialDataset
from lib.core.data import CoreDataModule

class DataModule(CoreDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def setup(self):
        self.train_dataset = MaterialDataset(self.cfg, 'train')
        self.val_dataset = MaterialDataset(self.cfg, 'test')
