import argparse
import os

import torch
from torch import nn, optim

os.makedirs("images", exist_ok=True)

IMG_SIZE = 128
LATENT_SIZE = 4
CHANNELS = 1
DEVICE = 'cuda'

class Reshape(nn.Module):
    '''
        Used in a nn.Sequential pipeline to reshape on the fly.
    '''

    def __init__(self, *target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(*self.target_shape)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.init_size = IMG_SIZE // 4
        self.l1 = nn.Sequential(nn.Linear(LATENT_SIZE, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, CHANNELS, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        # The height and width of downsampled image
        ds_size = IMG_SIZE // 2 ** 4
        self.model = nn.Sequential(
            *encoder_block(CHANNELS, 16, bn=False),
            *encoder_block(16, 32),
            *encoder_block(32, 64),
            *encoder_block(64, 128),
            Reshape([-1, ds_size]),
            nn.Linear(in_features=ds_size, out_features=IMG_SIZE),
            nn.LeakyReLU(),
            nn.Linear(in_features=IMG_SIZE, out_features=LATENT_SIZE)
        )


    def forward(self, img):
        return self.model(img)


def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()


def loss_function(pred, true, latent):
    return (pred-true).pow(2).mean(), MMD(torch.randn(200, LATENT_SIZE, requires_grad = False).to(DEVICE), latent)

class MMD_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, X):
        if self.training:
            latent = self.encoder(X)
            return self.decoder(latent), latent
        else:
            return self.decoder(self.encoder(X))

    def get_prob(self, frontier):
        # TODO: implement
        latent = self.encoder(torch.tensor(frontier.task.highresolution, dtype=torch.float, device='cuda'))
        return 1

    @staticmethod
    def train_model(net, learning_rate, epochs, train_loader, optimizer='Adam'):
        if optimizer == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                                  momentum=.95, nesterov=True)

        for epoch in range(epochs):
            training_loss = 0
            training_reconstruction_error = 0
            training_mmd = 0

            net.train()
            for batchnum, X in enumerate(train_loader):
                optimizer.zero_grad()

                X = X.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
                reconstruction, mu = net(X)
                reconstruction_error, mmd = loss_function(reconstruction, X, mu)
                loss = reconstruction_error + mmd
                loss.backward()

                optimizer.step()

                training_reconstruction_error += reconstruction_error
                training_mmd += mmd
                training_loss += loss

            training_reconstruction_error /= (batchnum + 1)
            training_mmd /= (batchnum + 1)
            training_loss /= (batchnum + 1)
            print('Training loss for epoch %i is %.8f, Reconstruction is %.8f, mmd is %.8f' % (
            epoch, training_loss, training_reconstruction_error, training_mmd))





#TODO save model; save training prob


