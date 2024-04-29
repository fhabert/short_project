import numpy as np
import matplotlib.pyplot as plt
import math
import torch 
import torch.nn as nn
import random
from PIL import Image
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import torch.optim as optim
import torchvision.utils as vutils
import os 

N = 3000
IMAGE_SIZE = 28
BATCH_SIZE = 32

categories = ["circle", "square", "star", "triangle"]
csv_file = "./data.csv"
manualSeed = 999
num_epochs = 5
nc = 1
ngf = 64
ndf = 64
lr = 0.001
data = []
nz = 100
ngpu = 1
beta1 = 0.5

def create_csv():
    for item in categories:
        for i in range(N):
            img = Image.open(f"./shapes/{item}/{i}.png")
            img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
            img_pixels = np.array(img) / 255
            data.append(img_pixels)

    np.random.shuffle(data)
    with open('data.csv', "w", newline="") as f:
        writer = csv.writer(f)
        for row in data:
            flattened_row = row.flatten()
            writer.writerow(flattened_row)
    pass

create_csv()

class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(1*IMAGE_SIZE * IMAGE_SIZE, ngf,device=device), # Grayscale => 1* imagesize * imagesize, RGB => 3* imagesize * imagesize
            nn.ReLU(),
            nn.Linear(ngf, ngf,device=device),
            nn.ReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(ngf, 16,device=device),
            nn.ReLU(),
            # Second hidden layer (16 -> output)
            nn.Linear(16, 1,device=device),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, header=None, sep=";", encoding="utf-8")  
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        sample = np.array(sample).reshape((28,28))
        #sample = Image.fromarray(sample)
        sample = torch.tensor(sample, dtype=torch.float32)
        #sample = self.transform(sample)
        return sample
        

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

custom_dataset = CustomDataset(csv_file, transform=transform)
dataloader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
   
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 4, 1, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 2, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
    
netG = Generator(ngpu).to(device)
netG.apply(weights_init)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(IMAGE_SIZE, nz, 1, 1, device=device)
real_label = 1.
fake_label = 0.
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

# print("Starting Training Loop...")
# for epoch in range(num_epochs):
#     for i, data in enumerate(dataloader, 0):
#         # Real images
#         netD.zero_grad()
#         real_cpu = data[0].to(device)
#         real_cpu = real_cpu.unsqueeze(0)
#         b_size = real_cpu.size(0)
#         label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
#         real_cpu = real_cpu.unsqueeze(0)
#         output = netD(real_cpu).view(-1)
#         print(output)
#         errD_real = criterion(output, label)
#         errD_real.backward()
#         D_x = output.mean().item()
#         # Fake images
#         noise = torch.randn(b_size, nz, 1, 1, device=device)
#         fake = netG(noise)
#         label.fill_(fake_label)
#         output = netD(fake.detach()).view(-1)
#         errD_fake = criterion(output, label)
#         errD_fake.backward()
#         D_G_z1 = output.mean().item()
#         errD = errD_real + errD_fake
#         optimizerD.step()

#         netG.zero_grad()
#         label.fill_(real_label)
#         output = netD(fake).view(-1)
#         errG = criterion(output, label)
#         errG.backward()
#         D_G_z2 = output.mean().item()
#         optimizerG.step()
#         if i % 50 == 0:
#             print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
#                   % (epoch, num_epochs, i, len(dataloader),
#                      errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

#         G_losses.append(errG.item())
#         D_losses.append(errD.item())
#         if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
#             with torch.no_grad():
#                 fake = netG(fixed_noise).detach().cpu()
#             img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

#         iters += 1
        

# fig = plt.figure(figsize=(8,8))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]

# def qft(qc):
#     for j in range(NUM_QUBITS):
#         for k in range(j):
#             qc.cry(np.pi/2, j,k)
#         qc.h(j)