import argparse
from omegaconf import OmegaConf
import torch
import torchvision.io as io
import torchvision.transforms as transforms
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
    parser.add_argument('--finetune', type=bool, required=False, default=True,
                        help='finetune for given example')
    parser.add_argument('--device', type=str, required=False, default='cuda',
                        help='use GPU / CPU')
    parser.add_argument('--h', type=int, required=False, default=384,
                        help='output height')
    parser.add_argument('--w', type=int, required=False, default=512,
                        help='output width')
    '''
    python scripts/test.py --model trainings/Neuralmaterial --test_image_id 0280
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

    weights_path = str(Path(args.model, 'checkpoint', 'latest.ckpt'))
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)

    # finetune model
    if args.finetune:
        weights_path = Path(args.model, 'checkpoint', f'{args.test_image_id}.ckpt')
        if not weights_path.is_file():
                trainer = Trainer(cfg.trainer)
                model = trainer.finetune(model, image, finetuning_steps)
                torch.save(model.state_dict(), weights_path)
        else:
            state_dict = torch.load(weights_path)
            model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    with torch.no_grad():
        # run forward pass and retrieve brdf decomposition
        image_out, brdf_maps, _, _ , _ = model.forward(image, 'test', size=(args.h, args.w))

    output_path = Path('outputs', 'resynthesis', args.test_image_id)
    output_path.mkdir(parents=True, exist_ok=True)

    # write outputs to disk
    io.write_png((image_out[0].clamp(0.0,1.0) * 255).byte().cpu(), str(Path(output_path,'rendering.png')))
    io.write_png((image[0].clamp(0.0,1.0) * 255).byte().cpu(), str(Path(output_path,'input.png')))

    for k, v in brdf_maps.items():

        if k == 'normal':
            v = (v + 1) / 2

        io.write_png((v[0].clamp(0.01, 0.99) * 255).byte().cpu(), str(Path(output_path, f'{k}.png')))

    print(f'synthesised {args.test_image_id}')
    
