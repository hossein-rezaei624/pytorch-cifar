import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import numpy as np
import matplotlib.pyplot as plt
import torchvision

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import random


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
  torch.cuda.manual_seed(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

transform_train = transforms.Compose([transforms.ToTensor(),])

trainset = torchvision.datasets.CIFAR100(root='./data', train = True, download=True, transform=transform_train)
filtered_indices = [idx for idx, (_, target, __) in enumerate(trainset) if target in [20, 90]]
filtered_data = torch.utils.data.Subset(trainset, filtered_indices)
trainloader = torch.utils.data.DataLoader(filtered_data, batch_size=len(filtered_indices), shuffle=False)

trainloader_ = torch.utils.data.DataLoader(filtered_data, batch_size=10, shuffle=True)

mapping = {value: index for index, value in enumerate([20, 90])}

net = ResNet18()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    confidence_epoch = []
    for batch_idx, (inputs, targets, indices_1) in enumerate(trainloader_):
        inputs, targets = inputs.to(device), targets.to(device)

        targets = torch.tensor([mapping[val.item()] for val in targets]).to(device)
      
        optimizer.zero_grad()
        outputs, soft_ = net(inputs)
        confidence_batch = []
        for i in range(targets.shape[0]):
          confidence_batch.append(soft_[i,targets[i]].item())
      
        # ... [Rest of the batch processing code]
        conf_tensor = torch.tensor(confidence_batch)
        Carto[epoch, indices_1] = conf_tensor  # Place confidences in the right location using indices

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader_), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
  


Carto = torch.zeros((4, len(trainset)))

for epoch in range(start_epoch, start_epoch+4):
    train(epoch)
    scheduler.step()


Confidence_mean = Carto.mean(dim=0)
Variability = Carto.std(dim=0)


plt.scatter(Variability, Confidence_mean, s = 2)

plt.xlabel("Variability") 
plt.ylabel("Confidence") 

plt.savefig('scatter_plot.png')


top_n = len(filtered_indices)

sorted_indices = np.argsort(Confidence_mean.numpy())

top_indices = sorted_indices[-top_n:]

top_indices = top_indices[::-1]

top_indices_sorted = top_indices[:75]


subset_data1 = torch.utils.data.Subset(trainset, top_indices_sorted)

# Extract the first 10 images
images1 = [subset_data1[i][0] for i in range(75)]
labels1 = [subset_data1[i][1] for i in range(75)]

# Make a grid from these images
grid = torchvision.utils.make_grid(images1, nrow=10)  # 5 images per row

torchvision.utils.save_image(grid, 'grid_image.png')


# Get all the filtered images and labels
images, labels, __ = next(iter(trainloader))

# Flatten the images
images_flat = images.view(images.shape[0], -1).numpy()
mean = np.mean(images_flat, axis=0)
std = np.std(images_flat, axis=0)
std_modified = np.where(std == 0, 1e-10, std)
images_normalized = (images_flat - mean) / std_modified

# Apply t-SNE
X_tsne = TSNE(n_components=2, random_state = 0).fit_transform(images_normalized)

# 3. Plot the results
plt.figure(figsize=(15, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="jet", edgecolor="None", alpha=0.4, s = 200)
plt.title('t-SNE - CIFAR10 Class 0 & 1')

# Annotate the points with their respective indices
for i, txt in enumerate(filtered_indices):
  if txt in top_indices_sorted:
    plt.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=12, alpha=0.9)

plt.savefig("tsne-image")
