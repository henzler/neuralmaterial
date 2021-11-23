import argparse
from omegaconf import OmegaConf
import torch
import torchvision.io as io
import torchvision.transforms as transforms
import numpy as np
import imageio
from pathlib import Path
import sys
sys.path.insert(0, './')

from lib.core.utils import seed_everything
from lib.core.trainer import Trainer
from lib.main import NeuralMaterial


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve BRDF decomposition from stationary flash image.')
    parser.add_argument('--model', type=str, required=True,
                        help='yaml config file')
    parser.add_argument('--test_image_id', type=str, required=True,
                        help='path to stationary flash image')
    parser.add_argument('--device', type=str, required=False, default='cuda',
                        help='use GPU / CPU')
    parser.add_argument('--h', type=int, required=False, default=384,
                        help='output height')
    parser.add_argument('--w', type=int, required=False, default=512,
                        help='output width')
    '''
    python scripts/relighting.py --model trainings/Neuralmaterial --test_image_id 0280
    '''

    finetuning_steps = 1000
    args = parser.parse_args()

    if torch.cuda.is_available() and args.device == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'

    # load config
    cfg = OmegaConf.load(str(Path(args.model, '.hydra', 'config.yaml')))
    seed_everything(cfg.seed)

    # load image
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((int(cfg.data.size[0]), int(cfg.data.size[1]))),
        transforms.ToTensor()]
    )

    # load all images in folder
    image_dirs = [str(p) for p in Path('flash_images', 'test', args.test_image_id).iterdir() if p.is_file()]

    # read first image in dir, change if required
    image = tfm(io.read_image(image_dirs[0]))[None]
    image = image.to(device)

    # load model with weights
    model = NeuralMaterial(cfg.model)

    ckpt_path = Path(args.model, 'checkpoint')
    fine_path = Path(ckpt_path, f'{args.test_image_id}.ckpt')

    if fine_path.is_file():
        weights_path = str(fine_path)
    else:
        weights_path = str(Path(args.model, 'checkpoint', 'latest.ckpt'))

    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # run forward pass and retrieve brdf decomposition
    _, brdf_maps, _, _ , _ = model.forward(image, 'test', size=(args.h, args.w))


    # rotate light along circle. Can be changed to anything you like
    images = []
    angles = torch.linspace(0, np.pi * 2, steps=60)

    for angle_i, angle in enumerate(angles):
        shift = torch.ones(1, 2, device=device)
        shift[:, 0] = 0.5 * np.cos(angle)
        shift[:, 1] = 0.5 * np.sin(angle)

        relit = model.renderer(brdf_maps, light_shift=shift)
        relit = (relit[0].cpu() * 255).byte().permute(1,2,0).numpy()
        images.append(relit)

    output_path = Path('outputs', 'relighting', args.test_image_id)
    output_path.mkdir(parents=True, exist_ok=True)

    imageio.mimwrite(str(Path(output_path, 'relit_circle.gif')), images, duration=0.05, loop=0)

    
