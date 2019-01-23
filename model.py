
# the paper defined hyper-parameter:chr
import numpy as np
import torch
from torch import nn

channel_rate = 64
# Note the image_shape must be multiple of patch_shape
image_shape = (256, 256, 3)
patch_shape = (channel_rate, channel_rate, 3)

ngf = 64
ndf = 64
input_nc = 3
output_nc = 3
input_shape_generator = (256, 256, input_nc)
input_shape_discriminator = (256, 256, output_nc)
n_blocks_gen = 9
use_bias = True
norm_layer = nn.BatchNorm2d
padding_type = 'reflect'
use_dropout = True
class Gnet(nn.Module):
    def __init__(self):
        super(Gnet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        n_downsampling = 2
        model = [self.layer1]
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                          stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks_gen):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

        # 03 changed
        self.combine = nn.Sequential(
            nn.Conv2d(output_nc*2, output_nc, kernel_size=7, padding=3)
        )

        # 04 changed
        self.combine_04 = nn.Sequential(
            nn.Conv2d(output_nc * 2, 8, kernel_size=7, padding=3),
            nn.Conv2d(8, 16, kernel_size=7, padding=3),
            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.Conv2d(16, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        )

        # 05 changed
        self.combine_05 = nn.Sequential(
            nn.Conv2d(output_nc * 2, 8, kernel_size=7, padding=3),
            nn.Conv2d(8, 16, kernel_size=7, padding=3),
            nn.Conv2d(16, output_nc, kernel_size=7, padding=3),
            nn.Sigmoid()
        )


    def forward(self, input):
        output = self.model(input)
        # output = output + input
        # output = output * 0.5
        # output = torch.clamp(output, min=0, max=1)
        output = torch.cat((input, output), 1)
        output = self.combine(output)
        output = torch.clamp(output, min=0, max=1)
        return output

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), #, dilation=1
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), #, dilation=1
                       norm_layer(dim)]
        # print(conv_block)
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_size=256, input_nc=input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        use_bias = True

        kw = 5
        padw = int(np.ceil((kw - 1) / 2))
        # print(padw)
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        # nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        size = int(input_size / 8)
        # sequence2 =[]
        # sequence2 += [
        #     nn.Flatten(),
        #     nn.Linear(size*size, 1024),
        #     nn.Tanh(),
        #     nn.Linear(1024, 1),
        #     nn.Sigmoid()
        #              ]
        self.model1 = nn.Sequential(*sequence)
        self.model2 = nn.Sequential(
            nn.Linear(size*size, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
                     )

    def forward(self, input):
        output = self.model1(input)
        # print('D1: ', output.shape)
        output = output.view(-1, int(output.shape[2]*output.shape[3]))
        output = self.model2(output)
        # print('D2: ', output.shape)
        return output


class generator_containing_discriminator(nn.Module):
    def __init__(self, Gnet, Dnet):
        super(generator_containing_discriminator, self).__init__()
        self.G = Gnet
        self.D =Dnet
    def forward(self, input):
        generated_image = self.G(input)
        output = self.D(generated_image)
        # print('G+D: ', output.shape)
        return output


class generator_containing_discriminator_multiple_outputs(nn.Module):
    def __init__(self, Gnet, Dnet):
        super(generator_containing_discriminator_multiple_outputs, self).__init__()
        self.G = Gnet
        self.D = Dnet
    def forward(self, input):
        generated_image = self.G(input)
        output = self.D(generated_image)
        return generated_image, output

if __name__ == '__main__':
    # g = Gnet()
    # img = np.zeros((1, 3, 100, 100))
    # img = torch.Tensor(img)
    # im = g(img)
    # print(im.shape)
    d = NLayerDiscriminator(input_nc=3)
    # print('-' * 5, 'Dnet', '-' * 5)
    # print(d)
    img = np.zeros((1, 3, 256, 256))
    img = torch.Tensor(img)
    im = d(img)
    print(im.shape)
    # m = generator_containing_discriminator(Gnet(), NLayerDiscriminator(input_nc=3))
    # # print('-' * 5, 'm', '-' * 5)
    # # print(m)
    # img = np.zeros((1, 3, 100, 100))
    # img = torch.Tensor(img)
    # im = m(img)
    # print(im.shape)
