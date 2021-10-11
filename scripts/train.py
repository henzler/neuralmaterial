from omegaconf import DictConfig
import os
import hydra

import sys
sys.path.insert(0, './')

from lib.core.trainer import Trainer
from lib.core.utils import seed_everything
from lib.main import NeuralMaterial
from lib.data import DataModule
from lib.logger import Logger


@hydra.main(config_path="../config", config_name="default.yaml")
def train(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))

    seed_everything(cfg.seed)

    logger = Logger()
    model = NeuralMaterial(cfg.model)
    data = DataModule(cfg.data)
    
    trainer = Trainer(cfg.trainer, logger)
    trainer.fit(model, data)

if __name__ == '__main__':
    train()
