from .losses import *
import torch
import kornia

class LossEngine(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.vgg = VGGFeatures()
        self.gram_loss = GramLoss()
        self.vggps_loss = VGGPSLoss()    

    def get_crops(self, image_in, image_out):

        bs, _, h, w = image_in.size()
        resample_size_h = h
        resample_size_w = w

        rand = torch.rand((1,)).item()
        start = self.cfg.crop[0]
        end = self.cfg.crop[1]
        zoom_factor = rand * (end - start) + start

        res = h * zoom_factor
        downscale = res / resample_size_h
        sigma = 2 * downscale / 6.0

        if zoom_factor > 1:
            image_in = kornia.filters.gaussian_blur2d(
                image_in, (5, 5), (sigma, sigma))
            image_out = kornia.filters.gaussian_blur2d(
                image_out, (5, 5), (sigma, sigma))

        grid = kornia.create_meshgrid(
            resample_size_h, resample_size_w,
            normalized_coordinates=True,
            device=torch.device(image_in.device)
        ).expand(bs, resample_size_h, resample_size_w, 2)

        grid = grid + 1 + torch.rand((1,)).item() * 2

        grid = (grid * zoom_factor) % 4.0
        grid = torch.where(grid > 2, 4 - grid, grid)
        grid = grid - 1

        crops_in = torch.nn.functional.grid_sample(
            image_in, grid, mode='bilinear', align_corners=True)

        crops_out = torch.nn.functional.grid_sample(
            image_out, grid, mode='bilinear', align_corners=True)

        return crops_in, crops_out

    def forward(self, image_in, image_out, mu, logvar, step):

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # slowly increase kl loss
        N = 10000
        if step < N:
            kl_loss = (step / N) * kl_loss

        crops_in, crops_out = self.get_crops(image_in, image_out)
        crops_in_vgg = self.vgg(crops_in)
        crops_out_vgg = self.vgg(crops_out)

        gram_loss = self.gram_loss(crops_out_vgg, crops_in_vgg)
        vggps_loss = self.vggps_loss(crops_out_vgg, crops_in_vgg)


        loss = gram_loss * self.cfg.gram + vggps_loss * self.cfg.vggps + self.cfg.kl * kl_loss

        losses = {
            'loss': loss.mean(),
            'gram': gram_loss,
            'vggps': vggps_loss,
            'kl': kl_loss,
        }
     
        return losses


