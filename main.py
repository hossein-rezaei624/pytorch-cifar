'''Train CIFAR10 with PyTorch.'''
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

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size_, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
net_ = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

net_ = net_.to(device)
if device == 'cuda':
    net_ = torch.nn.DataParallel(net_)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

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

          if indices_1[i] == 0:
            print(soft_[i,targets[i]].item())
      
##        if (targets.shape[0] != batch_size_):
##          for j in range(batch_size_ - targets.shape[0]):
##            confidence_batch.append(0)
        ##confidence_epoch.append(confidence_batch)
        #print(len(confidence_epoch[0]))


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
    
    ##conf_tensor = torch.tensor(confidence_epoch)
    ##conf_tensor = conf_tensor.reshape(conf_tensor.shape[0]*conf_tensor.shape[1])
    ##conf_tensor = conf_tensor[:(total-1)]
    #print(conf_tensor.shape)
    ##return conf_tensor

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

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

##Carto = []
Carto = torch.zeros((6, len(trainset)))
for epoch in range(start_epoch, start_epoch+6):
    ##Carto.append(train(epoch).numpy())
    train(epoch)
    test(epoch)
    scheduler.step()

print(Carto[:,0])


##Carto_tensor = torch.tensor(np.array(Carto))
#print(Carto_tensor.shape)
##Confidence_mean = Carto_tensor.mean(dim=0)
##Variability = Carto_tensor.std(dim = 0)
#print(Confidence_mean.shape)
#print(Variability.shape)

Confidence_mean = Carto.mean(dim=0)
Variability = Carto.std(dim=0)

plt.scatter(Variability, Confidence_mean, s = 2)


# Add Axes Labels

plt.xlabel("Variability") 
plt.ylabel("Confidence") 

# Display

plt.savefig('scatter_plot.png')


# Number of top values you're interested in
top_n = Variability.shape[0]//3

# Find the indices that would sort the array
sorted_indices = np.argsort(Variability.numpy())

# Take the last 'top_n' indices (i.e., the top values)
top_indices = sorted_indices[-top_n:]

top_indices = top_indices[::-1]

# If you want these indices in ascending order, you can sort them
#top_indices_sorted = np.sort(top_indices)

top_indices_sorted = top_indices

print(top_indices_sorted, top_indices_sorted.shape)

subset_data = torch.utils.data.Subset(trainset, top_indices_sorted)
trainloader_ = torch.utils.data.DataLoader(subset_data, batch_size=128, shuffle=True)


# Extract the first 10 images
images = [subset_data[i][0] for i in range(225)]
labels = [subset_data[i][1] for i in range(225)]

# Make a grid from these images
grid = torchvision.utils.make_grid(images, nrow=15)  # 5 images per row

# Print the labels (you can map these to actual class names if needed)
print("Labels:", labels)

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


for epoch in range(start_epoch, start_epoch+20):
    train_(epoch)
    test_(epoch)
    scheduler_.step()
