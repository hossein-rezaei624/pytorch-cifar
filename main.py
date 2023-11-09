import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse

from models import ResNet18
from utils import progress_bar

import numpy as np
import matplotlib.pyplot as plt
import torchvision

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

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

batch_size_ = 128

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size_, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=0)


# Model
print('==> Building model..')
net = ResNet18()
net_ = ResNet18()
net = net.to(device)
net_ = net_.to(device)

criterion = nn.CrossEntropyLoss()
criterion_ = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
optimizer_ = optim.SGD(net_.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    confidence_epoch = []
    for batch_idx, (inputs, targets, indices_1) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
  

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indices_1) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, __ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

  

Carto = torch.zeros((6, len(trainset)))

for epoch in range(start_epoch, start_epoch+6):
    train(epoch)
    test(epoch)
    scheduler.step()


Confidence_mean = Carto.mean(dim=0)
Variability = Carto.std(dim=0)


plt.scatter(Variability, Confidence_mean, s = 2)

plt.xlabel("Variability") 
plt.ylabel("Confidence") 

plt.savefig('scatter_plot.png')


top_n = Variability.shape[0]//4

sorted_indices = np.argsort(Confidence_mean.numpy())

top_indices = sorted_indices[:top_n]

#top_indices = top_indices[::-1]

top_indices_sorted = top_indices


subset_data = torch.utils.data.Subset(trainset, top_indices_sorted)
trainloader_ = torch.utils.data.DataLoader(subset_data, batch_size=128, shuffle=True)


# Extract the first 10 images
images = [subset_data[i][0] for i in range(15)]
labels = [subset_data[i][1] for i in range(15)]

# Make a grid from these images
grid = torchvision.utils.make_grid(images, nrow=15)  # 5 images per row

torchvision.utils.save_image(grid, 'grid_image.png')



def train_(epoch):
  net_.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets, indices_1) in enumerate(trainloader_):
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer_.zero_grad()
      outputs, soft_ = net_(inputs)

      loss = criterion_(outputs, targets)
      loss.backward()
      optimizer_.step()

      train_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      progress_bar(batch_idx, len(trainloader_), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    

def test_(epoch):
    net_.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indices_1) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, __ = net_(inputs)
            loss = criterion_(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print("\n")

print("Trainning with cartography...")
for epoch in range(start_epoch, start_epoch+20):
    train_(epoch)
    test_(epoch)
    scheduler_.step()
