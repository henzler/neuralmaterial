import torch
import torch.nn as nn
import kornia

class Decoder(nn.Module):
    def __init__(self, w, z, layers):
        super(Decoder, self).__init__()

        self.layers = layers
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.down_mappings = nn.ModuleList()
        self.up_mappings = nn.ModuleList()
        self.w = w

        self.init_mapping = Mapping(z, w)

        self.adain = AdaptiveInstanceNormalization()

        for i in range(layers):

            f_in = int(min(w * (2 ** i), 256))
            f_out = int(min(f_in * 2, 256))

            self.down_mappings.append(Mapping(z, f_out))
            self.down_blocks.append(StyleBlock(f_in, f_out))

        self.bottom_mapping = Mapping(z, f_out)
        self.bottom_conv = StyleBlock(f_out, f_out)

        for i in range(layers):
            f_in = int(min(w * (2 ** (layers - i)), 256))
            f_out = int(min(w * (2 ** (layers - i - 1)), 256))

            self.up_mappings.append(Mapping(z, f_out))
            self.up_blocks.append(StyleBlock(f_in, f_out))

        self.final = nn.Sequential(
            nn.Conv2d(f_out, 6, 1, 1, 0)
        )

        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, z, x):

        mean, var = self.init_mapping(z)
        x = self.adain(x, mean, var)

        down = []

        for i in range(self.layers):
            mean, var = self.down_mappings[i](z)
            x = self.down_blocks[i](x, mean, var)
            down.append(x)
            x = self.max_pool(x)

        mean, var = self.bottom_mapping(z)
        x = self.bottom_conv(x, mean, var)

        for j in range(self.layers):

            x = nn.functional.interpolate(x, mode='nearest', scale_factor=2)
            x = kornia.gaussian_blur2d(x, (3, 3), (0.2, 0.2), 'circular')

            skip_x = down[-(j+1)]
            mean, var = self.up_mappings[j](z)
            x = self.up_blocks[j](x + skip_x, mean, var)

        x = self.final(x)
        x = torch.sigmoid(x)

        return x

class StyleBlock(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()

        self.conv = nn.Conv2d(in_f, out_f, 3, 1, 1, padding_mode='circular')
        self.adain = AdaptiveInstanceNormalization()
        self.lrelu = nn.LeakyReLU()

    def forward(self, x, mean, var):

        x = self.conv(x)
        x = self.adain(x, mean, var)
        x = self.lrelu(x)

        return x

class Mapping(nn.Module):
    def __init__(self, z_size, out_size):
        super(Mapping, self).__init__()

        self.out_size = out_size

        self.mapping_layers = nn.ModuleList()
        self.linear = nn.Linear(z_size, z_size)
        self.relu = nn.ReLU(inplace=True)

        self.affine_transform = nn.Linear(z_size, out_size * 2)
        self.affine_transform.bias.data[:out_size] = 0
        self.affine_transform.bias.data[out_size:] = 1

    def forward(self, z):

        z = self.relu(self.linear(z))
        x = self.affine_transform(z)
        mean, std = torch.split(
            x, [self.out_size, self.out_size],
            dim=1
        )

        mean = mean[..., None, None]
        std = std[..., None, None]

        return mean, std

class AdaptiveInstanceNormalization(nn.Module):
    def and__init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()

    def forward(self, x, mean, std):
        whitened_x = torch.nn.functional.instance_norm(x)
        return whitened_x * std + mean