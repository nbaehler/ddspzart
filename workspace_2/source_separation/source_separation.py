import openunmix
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch
import torch.utils
import os
import scipy.signal
import numpy as np
from tqdm import tnrange, tqdm_notebook
import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings
import random
import sklearn.preprocessing
import norbert
import musdb
warnings.simplefilter(action='ignore', category=FutureWarning)


os.chdir("../open-unmix-pytorch")


class SimpleMUSDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        subset='train',
        split='train',
        target='vocals',
        seq_duration=None,
    ):
        """MUSDB18 Dataset wrapper
        """
        self.seq_duration = seq_duration
        self.target = target
        self.mus = musdb.DB(
            download=True,
            split=split,
            subsets=subset,
        )

    def __getitem__(self, index):
        track = self.mus[index]
        track.chunk_start = random.uniform(
            0, track.duration - self.seq_duration)
        track.chunk_duration = self.seq_duration
        x = track.audio.T
        y = track.targets[self.target].audio.T
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.mus)


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def main():
    train_dataset = SimpleMUSDBDataset(seq_duration=5.0)
    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True)
    print(len(train_dataset))

    # create a spectrogram layer
    # -> shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
    stft = openunmix.transforms.TorchSTFT()
    # -> shape (nb_samples, nb_channels(=1 if mono), nb_bins, nb_frames)
    spec = openunmix.transforms.ComplexNorm(mono=True)
    transform = nn.Sequential(stft, spec)

    x, y = train_dataset[7]
    print(x.shape)

    # transform the time domain input to spectrograms
    X = transform(x[None])
    Y = transform(y[None])
    print(X.shape)

    f, axes = plt.subplots(1, 2)
    axes[0].pcolormesh(np.log(X[0, 0, :, :].detach().numpy()))
    axes[1].pcolormesh(np.log(Y[0, 0, :, :].detach().numpy()))

    scaler = sklearn.preprocessing.StandardScaler()

    for x, y in tqdm.tqdm_notebook(train_dataset):
        X = transform(x[None]).T
        scaler.partial_fit(X.squeeze().numpy())

    # set initial input scaler values
    scale = np.maximum(
        scaler.scale_,
        1e-4*np.max(scaler.scale_)
    )
    mean = scaler.mean_

    f, axes = plt.subplots(2, 1)
    axes[0].plot(mean)
    axes[1].plot(scale)

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    unmix = openunmix.model.OpenUnmix(
        input_mean=mean,
        input_scale=scale,
        nb_channels=1,
        hidden_size=256,
        max_bin=64,
        nb_bins=2048+1
    ).to(device)

    optimizer = optim.RMSprop(unmix.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    losses = openunmix.utils.AverageMeter()
    unmix.train()

    for i in tqdm.tqdm_notebook(range(1)):
        for x, y in tqdm.tqdm_notebook(train_sampler):
            x, y = x.to(device), y.to(device)
            X = transform(x)
            Y = transform(y)
            optimizer.zero_grad()
            Y_hat = unmix(X)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), Y.size(1))
        print(losses.avg)

    mus = musdb.DB(download=True, subsets='test')
    track = mus[1]

    unmix.eval()
    x, y = x.to(device), y.to(device)
    X = transform(x)
    Y = transform(y)
    optimizer.zero_grad()
    Y_hat = unmix(X)

    audio_hat = istft(
        Y[..., j].T,
        n_fft=unmix.stft.n_fft,
        n_hopsize=unmix.stft.n_hop
    )

    target_models = {"vocals": unmix}
    separator = openunmix.model.Separator(target_models)
    audio_torch = torch.tensor(track.audio.T[None, ...]).float().to(device)
    print(audio_torch.shape)
    separator.forward(audio_torch)


if __name__ == "__main__":
    main()
