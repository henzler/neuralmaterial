import torch.nn as nn
import torch
import torch.fft
import lib.models.vgg as vgg


class VGGFeatures(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

        model = vgg.vgg19(True, 3)
        model.eval()
        vgg_pretrained_features = model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(4):  # relu_1_1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):  # relu_2_1
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):  # relu_3_1
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):  # relu_4_1
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):  # relu_5_1
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):

        ## normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        device = 'cuda' if x.is_cuda else 'cpu'
        mean = torch.as_tensor(mean, device=device).view(1, 3, 1, 1)
        std = torch.as_tensor(std, device=device).view(1, 3, 1, 1)
        x = x.sub(mean)
        x = x.div(std)

        # get features
        h1 = self.slice1(x)
        h_relu1_1 = h1
        h2 = self.slice2(h1)
        h_relu2_1 = h2
        h3 = self.slice3(h2)
        h_relu3_1 = h3
        h4 = self.slice4(h3)
        h_relu4_1 = h4
        h5 = self.slice5(h4)
        h_relu5_1 = h5

        return [h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1]

class GramMatrix(torch.nn.Module):

    def forward(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        gram_matrix = torch.bmm(features, features.transpose(1, 2))
        gram_matrix.div_(h * w)
        return gram_matrix

class VGGMeanLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1_loss = torch.nn.L1Loss()

    def forward(self, x, y):
        loss = torch.tensor(0.0, device=x[0].device)

        input_features = x
        output_features = y

        for idx, (input_feature, output_feature) in enumerate(
                zip(input_features, output_features)):

            bs,c,h,w = input_feature.size()
            input_feature = input_feature.reshape(bs,c,h*w).mean(dim=2)
            output_feature = output_feature.reshape(bs,c,h*w).mean(dim=2)

            loss += self.l1_loss(input_feature, output_feature).mean()

        return loss

class SWDLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.mse_loss = torch.nn.MSELoss()

    def forward(self, xs, ys):
        device = xs[0].device
        loss = torch.tensor(0.0, device=device)

        for idx, (x, y) in enumerate(
                zip(xs, ys)):

            bs, c, h, w = x.size()

            x = x.reshape(bs, c, h * w)
            y = y.reshape(bs, c, h * w)

            direction = torch.torch.randn(bs, c, c, device=device)
            direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True)

            proj_x = torch.bmm(direction, x)
            proj_y = torch.bmm(direction, y)

            proj_x, _ = torch.sort(proj_x, dim=-1)
            proj_y, _ = torch.sort(proj_y, dim=-1)

            loss += self.mse_loss(proj_x, proj_y).mean()

        return loss


class PSLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1_loss = torch.nn.L1Loss()

    def forward(self, x, y):

        x_power = torch.abs(torch.fft.fftn(x, dim=[2, 3]))
        y_power = torch.abs(torch.fft.fftn(y, dim=[2, 3]))

        loss = self.l1_loss(x_power, y_power).sum()

        return loss

class VGGLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x, y):
        loss = torch.tensor(0.0, device=x[0].device)

        input_features = x
        output_features = y

        for idx, (input_feature, output_feature) in enumerate(
                zip(input_features, output_features)):

            loss += self.mse_loss(output_feature, input_feature).mean()

        return loss


class GramLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.gram_matrix = GramMatrix()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, x, y):
        loss = torch.tensor(0.0, device=x[0].device)

        input_features = x
        output_features = y

        for idx, (input_feature, output_feature) in enumerate(
                zip(input_features, output_features)):
            gram_out = self.gram_matrix(output_feature)
            gram_in = self.gram_matrix(input_feature)
            loss += self.l1_loss(gram_out, gram_in).mean()

        return loss



class VGGPSLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1_loss = torch.nn.L1Loss()

    def forward(self, x, y):
        loss = torch.tensor(0.0, device=x[0].device)

        features_out = x
        features_gt = y

        for idx, (feature_out, feature_gt) in enumerate(
                zip(features_out, features_gt)):

            x_power = torch.abs(torch.fft.fftn(feature_out, dim=[2, 3]))
            y_power = torch.abs(torch.fft.fftn(feature_gt, dim=[2, 3]))

            loss += self.l1_loss(x_power, y_power).sum()

        return loss

class VGGPSLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1_loss = torch.nn.L1Loss()

    def forward(self, x, y):
        loss = torch.tensor(0.0, device=x[0].device)

        features_out = x
        features_gt = y

        for idx, (feature_out, feature_gt) in enumerate(
                zip(features_out, features_gt)):

            x_power = torch.abs(torch.fft.fftn(feature_out, dim=[2, 3]))
            y_power = torch.abs(torch.fft.fftn(feature_gt, dim=[2, 3]))

            loss += self.l1_loss(x_power, y_power).sum()

        return loss