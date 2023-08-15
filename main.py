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

class_sets = [
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



transform11 = nn.Sequential(
    RandomResizedCrop(size = (32,32), scale=(0.2, 1.)),
    RandomHorizontalFlip(),
    ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
    RandomGrayscale(p=0.2)
)

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = Reduced_ResNet18(100)
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

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      weight_decay=0)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs = transform11(inputs)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))




def test(epoch, class_sets):

  with torch.no_grad():
      for i, class_set in enumerate(class_sets):
          correct = 0
          total = 0
  
          for images, labels in test_loader:
              images = images.to(device)
              labels = labels.to(device)
  
              # Get the predicted outputs from the model
              outputs = net(images)
  
              # Select only the predictions corresponding to the current class set
              selected_outputs = outputs[:, class_set]
  
              # Get the predicted class indices
              _, predicted = torch.max(selected_outputs.data, 1)
  
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
  
          accuracy = 100 * correct / total
          print(f"Accuracy for Set {i}: {accuracy:.2f}%")


test(1, class_sets)
