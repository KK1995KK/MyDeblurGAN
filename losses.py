import torch
from torch import nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        self.model = nn.Sequential()
        for i, layer in enumerate(list(cnn)):
            self.model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        # print(self.model)
        # print(cnn)

    def forward(self, x):
        output = self.model(x)
        return output


class PerceptualLoss_v2(nn.Module):
    def __init__(self):
        super(PerceptualLoss_v2, self).__init__()

    def initialize(self, loss):
        self.criterion = loss
        self.contentFunc = Encoder().double().cuda()

    def forward(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad).mean()
        # loss = torch.mean(torch.square(f_real_no_grad - f_fake))
        # print(loss)
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

    def initialize(self, loss):
        self.criterion = loss
        # self.contentFunc = Encoder().double().cuda()

    def forward(self, fakeIm, realIm):
        # f_fake = self.contentFunc.forward(fakeIm)
        # f_real = self.contentFunc.forward(realIm)
        # f_real_no_grad = f_real.detach()
        # loss = self.criterion(f_fake, f_real_no_grad)
        # loss = torch.mean(torch.square(f_real_no_grad-f_fake))
        loss = self.criterion(fakeIm, realIm)
        return loss


class WassesrsteinLoss(nn.Module):
    def __init__(self):
        super(WassesrsteinLoss, self).__init__()

    def forward(self, fakeIm, realIm):
        loss1 = abs(fakeIm - realIm)
        # loss1 = fakeIm * realIm
        return loss1.mean()
        # return torch.mean(torch.mm(fakeIm, realIm))
        # return (fakeIm*realIm).mean()

if __name__=='__main__':
    contentFunc = Encoder()
