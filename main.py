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
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale


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

sets = [
    [26, 86, 2, 55, 75, 93, 16, 73, 54, 95],
    [53, 92, 78, 13, 7, 30, 22, 24, 33, 8],
    [43, 62, 3, 71, 45, 48, 6, 99, 82, 76],
    [60, 80, 90, 68, 51, 27, 18, 56, 63, 74],
    [1, 61, 42, 41, 4, 15, 17, 40, 38, 5],
    [91, 59, 0, 34, 28, 50, 11, 35, 23, 52],
    [10, 31, 66, 57, 79, 85, 32, 84, 14, 89],
    [19, 29, 49, 97, 98, 69, 20, 94, 72, 77],
    [25, 37, 81, 46, 39, 65, 58, 12, 88, 70],
    [87, 36, 21, 83, 9, 96, 67, 64, 47, 44]
]



transform_test = transforms.Compose([
    transforms.ToTensor(),
])


# Model
print('==> Building model..')
# net = VGG('VGG19')
net = Reduced_ResNet18(100)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      weight_decay=0)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])




dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

# Iterate over each set
for i, subset_classes in enumerate(sets):
    # Get indices of samples belonging to the current subset
    subset_indices = [idx for idx, (_, target) in enumerate(dataset) if target in subset_classes]

    # Create a data loader for the current subset
    subset_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, subset_indices), batch_size=64)

    correct = 0
    total = 0

    # Calculate accuracy for the current subset
    with torch.no_grad():
        for data in subset_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Set: {i}, Accuracy: {100 * correct / total}%')
