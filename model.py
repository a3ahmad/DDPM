import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as c
from torch.utils.data import DataLoader
from torchvision.models import vgg

from pytorch_lightning.core.lightning import LightningModule

import math
import numpy as np

from PIL import Image


class Upsample(nn.Module):
    def __init__(self, factor):
        super(Upsample, self).__init__()

        self.factor = factor

    def forward(self, x):
        x = nn.functional.interpolate(
            x,
            scale_factor=self.factor,
            mode='bilinear',
            align_corners=False)
        return x


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# From the PyTorch docs on Transformers
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        d_model = d_model // 2
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe = torch.cat([torch.sin(position * div_term), torch.cos(position * div_term)], dim=1)
        self.register_buffer('pe', pe)  

    def forward(self, t):
        return self.pe[t]


class ConvBlock(nn.Module):
    def __init__(self, inDim, outDim, downsample=False, dropout=0.1):
        super(ConvBlock, self).__init__()

        self.norm = nn.GroupNorm(32, inDim)
        self.act = Mish()
        self.dropout = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(inDim, outDim, kernel_size=3, padding=1, stride=2 if downsample else 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class EncodingBlock(nn.Module):
    def __init__(self, args, inDim, outDim, width=4, downsample=False):
        super(EncodingBlock, self).__init__()

        self.inDim = inDim
        self.outDim = outDim
        self.downsample = downsample

        self.block1 = ConvBlock(inDim, width * inDim)
        self.pos_enc = nn.Sequential(
            nn.Linear(args.T, width * inDim),
            Mish(),
            nn.Linear(width * inDim, width * inDim),
        )
        self.block2 = ConvBlock(width * inDim, outDim, downsample)

    def forward(self, x, t):
        h = self.block1(x)
        h = h + self.pos_enc(t).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)

        return x + h if self.inDim == self.outDim and not self.downsample else h


class DecodingBlock(nn.Module):
    def __init__(self, args, inDim, outDim, width=4, upsample=False):
        super(DecodingBlock, self).__init__()

        self.inDim = inDim
        self.outDim = outDim
        self.upsample = upsample

        self.block1 = ConvBlock(inDim, width * inDim)
        self.pos_enc = nn.Sequential(
            nn.Linear(args.T, width * inDim),
            Mish(),
            nn.Linear(width * inDim, width * inDim),
        )
        self.block2 = ConvBlock(width * inDim, outDim)
        self.up2x = Upsample(2.0)

    def forward(self, x, t):
        h = self.block1(x)
        h = h + self.pos_enc(t).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)

        result = x + h if self.inDim == self.outDim else h

        return self.up2x(result) if self.upsample else result


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()

        self.encoders = nn.ModuleList([
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            EncodingBlock(args, 128, 128, downsample=True),
            EncodingBlock(args, 128, 256, downsample=True),
            EncodingBlock(args, 256, 256, downsample=True),
            EncodingBlock(args, 256, 512, downsample=True),
            EncodingBlock(args, 512, 512, downsample=True),
            EncodingBlock(args, 512, 512, downsample=True),
        ])

        self.decoders = nn.ModuleList([
            DecodingBlock(args, 512, 512, upsample=True),
            DecodingBlock(args, 2*512, 512, upsample=True),
            DecodingBlock(args, 2*512, 256, upsample=True),
            DecodingBlock(args, 2*256, 256, upsample=True),
            DecodingBlock(args, 2*256, 128, upsample=True),
            DecodingBlock(args, 2*128, 128, upsample=True),
            nn.Conv2d(2*128, 3, kernel_size=3, padding=1),
        ])

        self.time_embedding = PositionalEncoding(args.T, max_len=args.T)


    def forward(self, x, t):
        t = self.time_embedding(t)
        
        h = x
        hs = []
        for _, enc in enumerate(self.encoders):
            h = enc(h, t) if isinstance(enc, EncodingBlock) else enc(h)
            hs.append(h)
        hs[-1] = None    # Drop the last one
        hs.reverse()
        
        for _, (dec, last_h) in enumerate(zip(self.decoders, hs)):
            h = torch.cat((h, last_h), dim=1) if last_h is not None else h
            h = dec(h, t) if isinstance(dec, DecodingBlock) else dec(h)

        return torch.sigmoid(h)


class DDPM(LightningModule):

    def __init__(self, args):
        super().__init__()

        self.args = args

        Bsched = torch.linspace(args.B1, args.BT, args.T)
        self.register_buffer("Bsched", Bsched)
        alpha = 1.0 - self.Bsched
        self.register_buffer("alpha", alpha)
        alpha_bar = torch.exp(torch.cumsum(torch.log(self.alpha), dim=0))
        #alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar).detach()
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar)
        one_minus_sqrt_alpha_bar = torch.sqrt(1.0 - self.alpha_bar).detach()
        self.register_buffer("one_minus_sqrt_alpha_bar", one_minus_sqrt_alpha_bar)
        noisesample = torch.empty((1, 3, 1024, 1024)).normal_()
        self.register_buffer("noisesample", noisesample)

        self.diffuser = UNet(args)


    def sample(self, noise):
        x = noise
        for t in reversed(range(self.args.T)):
            z = torch.randn_like(noise) if t > 0 else torch.zeros_like(noise)
            a = self.alpha[t]
            ab = self.alpha_bar[t]
            b = self.Bsched[t]
            x = (x - (1.0 - a) * self.diffuser(x, t) / torch.sqrt(1.0 - ab)) / torch.sqrt(a) + z * torch.sqrt(b)
        return x

    def forward(self, img):
        noise = torch.randn_like(img)
        t = torch.randint(self.args.T, (1,), device=img.device)
        diffusion = self.sqrt_alpha_bar[t] * img + self.one_minus_sqrt_alpha_bar[t] * noise
        diffused = self.diffuser(diffusion.type(torch.cuda.HalfTensor), t)
        return noise, diffused

    def training_step(self, batch, batch_idx):
        noise, diffused = self.forward(batch)
        loss = F.mse_loss(noise, diffused)
        #loss = torch.mean((noise - diffused).view(img.shape[0], -1)**2, dim=1).mean()
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        with torch.no_grad():
            img = self.noisesample
            img = np.squeeze(((img + 1.0) * 127.5).cpu().numpy().astype(np.uint8))
            img = np.transpose(img, (1, 2, 0))
            im = Image.fromarray(img, 'RGB')
            im.save('noisesample.png')

            img = self.sample(self.noisesample)
            img = np.squeeze(((img + 1.0) * 127.5).cpu().numpy().astype(np.uint8))
            img = np.transpose(img, (1, 2, 0))
            im = Image.fromarray(img, 'RGB')
            im.save('sample.png')
        return {}

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.args.lr)
        return opt
