from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.io as io
import json
import torch

class MaterialDataset(Dataset):

    def __init__(self, cfg, mode):

        self.cfg = cfg
        self.path_dir = Path(cfg.path)

        self.mode = mode
        self.samples = self.get_samples()

        if len(self.samples) == 0:
            exit("Could not find data samples")

        self.dataset_length = len(self.samples)

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                (int(self.cfg.size[0]), int(self.cfg.size[1]))),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()])

    def __len__(self):
        return self.dataset_length

    def get_samples(self):

        with open(str(Path(self.path_dir, 'metadata.json')), 'r') as f:
            metadata = json.load(f)

        materials = metadata['materials']

        samples = []
        for material in materials:

            if material['split'] != self.mode:
                continue

            for entry in material['entries']:
                entry['id'] = material['material_id']
                entry['split'] = material['split']
                samples.append(entry)

        return samples

    def __getitem__(self, idx):

        sample = self.samples[idx]
        filepath = Path(
            self.path_dir, sample['split'], sample['id'],
            f"{sample['name']}{sample['suffix']}"
        )

        image = io.read_image(str(filepath))
        image = self.transforms(image)
        shuffle_idx = torch.randperm(3)
        image = image[shuffle_idx, ...]

        return image
