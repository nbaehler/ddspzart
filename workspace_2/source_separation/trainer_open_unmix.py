
from turtle import forward
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import openunmix
import torch
from torch.optim.lr_scheduler import StepLR


class SourceSeparation():

    def __init__(self, mean, scale) -> None:
        # -> shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
        self.stft = openunmix.transforms.TorchSTFT()
        # -> shape (nb_samples, nb_channels(=1 if mono), nb_bins, nb_frames)
        self.spec = openunmix.transforms.ComplexNorm(mono=True)
        self.transform = nn.Sequential(self.stft, self.spec)

        self.unmix = openunmix.model.OpenUnmix(
            input_mean=mean,
            input_scale=scale,
            nb_channels=1,
            hidden_size=512,
            max_bin=512,
            nb_bins=2048+1
        )

    def forward(self, x):
        x_ = self.transform(x)
        x_ = self.unmix(x_)
        return x_

    def __call__(self, x):
        return self.forward(x)

    def forward_gt(self, y):
        return self.transform(y)


class SourceSeperationLight(pl.LightningModule):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.losses = openunmix.utils.AverageMeter()
        self.losses_valid = openunmix.utils.AverageMeter()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        Y_hat = self.model(x)
        Y = self.model.forward_gt(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            self.model.unmix.parameters(), lr=0.005)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        Y_hat = self.model(x)
        Y = self.model.forward_gt(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        Y_hat = self.model(x)
        Y = self.model.forward_gt(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss
