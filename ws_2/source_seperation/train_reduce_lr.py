

#import scipy.signal
print('Start')
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import warnings
import random
import sklearn.preprocessing
import norbert
import musdb
import torch
import os
import pickle 
import openunmix
import torch.nn as nn
import time
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from dataloader_slakh import SlakhDataset
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    
    VALID_EVERY_N_EPOCH = 5
    TARGET = "Trumpet" #"Electric Guitar (jazz)" #"Flute"
    NUM_EPOCH = 300
    CKPT_PATH = f"../source_seperation/data/checkpoints/exp09_cluster_trumpet_v5.pt"
    #CKPT_LOAD_PATH = f"../source_seperation/data/checkpoints/exp8_cluster_flute.pt"
    
    writer = SummaryWriter(log_dir="../source_seperation/runs/exp09_cluster_trumpet_v5")
    
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    pin_memory = True if use_cuda else False

    #device = "cpu" #FIXME Remove
    
    print(TARGET)
    print(CKPT_PATH)
    print(device)
    
    
    tic = time.perf_counter()
    
    print("Get datasets")
    train_dataset = SlakhDataset(target=TARGET, seq_duration=5.0)
    print(train_dataset.targets[0])
    print("length", len(train_dataset))

    validation_dataset = SlakhDataset(target=TARGET,split='validation',seq_duration=5.0)
    test_dataset = SlakhDataset(target=TARGET,split='test',seq_duration=5.0)
    
    train_sampler = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=pin_memory)
    validation_sampler = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=pin_memory)
    
    test_sampler = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True,pin_memory=pin_memory,num_workers=4)

    stft = openunmix.transforms.TorchSTFT() # -> shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
    spec = openunmix.transforms.ComplexNorm(mono=True) # -> shape (nb_samples, nb_channels(=1 if mono), nb_bins, nb_frames)
    transform = nn.Sequential(stft, spec)
  
    toc = time.perf_counter()
    print("Needed time: ", toc-tic)
    
    print("Get scaling")
    scaler = sklearn.preprocessing.StandardScaler()

    for x, y in train_dataset:
        X = transform(x[None]).T
        scaler.partial_fit(X.squeeze().numpy())

    # set inital input scaler values
    scale = np.maximum(
        scaler.scale_,
        1e-4*np.max(scaler.scale_)
    )
    mean = scaler.mean_
    
    print("Needed time: ", toc-tic)

    print("Build model")
    unmix = openunmix.model.OpenUnmix(
        input_mean=mean,
        input_scale=scale,
        nb_channels=1,
        hidden_size=512,
        max_bin=512,
        nb_bins=2048+1
    ).to(device)

    optimizer = optim.RMSprop(unmix.parameters(), lr=0.06)
    #optimizer = optim.Adam(unmix.parameters(), lr=0.06)
    
    criterion = torch.nn.MSELoss()
    
    
    epoch=0
    #checkpoint = torch.load(CKPT_LOAD_PATH)
    #unmix.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss'] 

    scheduler = StepLR(optimizer, step_size=30, gamma=0.6)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.25,patience=30, verbose=True)

    stft = openunmix.transforms.TorchSTFT() 
    spec = openunmix.transforms.ComplexNorm(mono=True)
    transform = nn.Sequential(stft, spec).to(device)

    print("Training")
    # Training
    losses = openunmix.utils.AverageMeter()
    losses_valid = openunmix.utils.AverageMeter()
    unmix.train()
    stft = openunmix.transforms.TorchSTFT() 
    spec = openunmix.transforms.ComplexNorm(mono=True)
    transform = nn.Sequential(stft, spec).to(device)

    keep_epoch = epoch

    
    for epoch in range(keep_epoch,NUM_EPOCH+keep_epoch):
        tic = time.perf_counter()
        tic2 = time.process_time()
        print("Epoch: ", epoch)
        for x, y in train_sampler: #tqdm.notebook.tqdm(train_sampler): 
            x, y = x.to(device), y.to(device)
            X = transform(x)
            Y = transform(y)
            optimizer.zero_grad()
            Y_hat = unmix(X)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), Y.size(1))
        writer.add_scalar("Loss/train", losses.avg, epoch)
        print(f"Loss={losses.avg:.3f}", "LR: ", scheduler.get_last_lr())
        keep_epoch = epoch + 1

        toc = time.perf_counter()
        toc2 = time.process_time()

        print("Time elpased for epoch", toc - tic, toc2 - tic2)

        if epoch % VALID_EVERY_N_EPOCH == 0:
            unmix.eval()
            with torch.no_grad():
                val_loss = []
                for x,y in validation_sampler:
                    x, y = x.to(device), y.to(device)
                    X = transform(x)
                    Y = transform(y)
                    Y_hat = unmix(X)
                    loss_valid = torch.nn.functional.mse_loss(Y_hat, Y)
                    losses_valid.update(loss_valid.item(), Y.size(1))
                writer.add_scalar("Loss/validation", losses_valid.avg, epoch)
                print(f"validation: {losses_valid.avg:.3f}")
            unmix.train()      
        
        scheduler.step()
        #scheduler.step(losses_valid.avg)     

    writer.flush()
    writer.close()

    torch.save({
            'epoch': epoch,
            'model_state_dict': unmix.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, CKPT_PATH)

    print("Saved model")

    print("testing...")
    # Testing

    for epoch in range(5):
        for x, y in test_sampler:
            x, y = x.to(device), y.to(device)
            X = transform(x)
            Y = transform(y)
            Y_hat = unmix(X)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))
        print(f"{losses.avg:.3f}")


if __name__ == "__main__":
    main()