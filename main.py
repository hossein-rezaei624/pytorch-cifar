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
from matplotlib import pyplot as plt
from skimage.util import random_noise

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
    transforms.RandomRotation(45),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
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
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


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

checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
###print('\n\nLayer params:')
tempp = 0
#weights_ = torch.zeros((512,10))
#bias_ = torch.zeros((10))
for param in net.parameters():
    tempp +=1
    if (tempp==109):
      ###print(param)
      ###print("the shapeeeeeee",param.shape)
      weights_ = param
    if (tempp==110):
      ###print(param)
      ###print("the shapeeeeeee",param.shape)
      bias_ = param
#print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",net.state_dict())
print("temppppppppp",tempp)
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    count = 0
    cc_ = []
    some_new = []
    some_new_1 = []
    some_new_2 = []
    some_new_3 = []
    some_new_4 = []
    some_accuracy = []
    max_1 = []
    max_2 = []
    with torch.no_grad():        
        counter = 0
        for batch_idx, (img, label) in enumerate(testloader):
          img, label = img.to(device), label.to(device)
          counter += 1
          temp11 = []
          temp22 = []
          temp33 = []
          temp44 = []
          #jitter = torchvision.transforms.ColorJitter(brightness=.5, hue=.3)
          #img = jitter(img)
          #img = torchvision.transforms.functional.adjust_brightness(img, brightness_factor = 1)
          #img = torchvision.transforms.functional.adjust_contrast(img, contrast_factor = 1)

          #img = torchvision.transforms.functional.rotate(img, 90)
          #img = torchvision.transforms.functional.gaussian_blur(img, kernel_size=(5, 9), sigma=(0.1, 5))
          #img = torchvision.transforms.functional.adjust_hue(img, hue_factor = 0.2)

          outputs, rep = net(img)
          loss = criterion(outputs, label)
          test_loss = loss.item()
          _, predicted = outputs.max(1)
          correct = predicted.eq(label).sum().item()


          a = rep
          b = weights_.transpose(0,1)
          final = torch.matmul(a,b)+bias_

          inner_product = torch.matmul(a,b)
          a_norm = a.pow(2).sum(dim=1).pow(0.5)
          b_norm = b.pow(2).sum(dim=0).pow(0.5)
          hh = torch.matmul(a_norm.view((a_norm.shape[0],1)),b_norm.view((1,10)))
          cos = inner_product / hh
          angle = (torch.acos(cos)*57.2958)

          cc = 0
          for h in range(label.shape[0]):
            if predicted[h] != label[h]:
              continue
            cc += 1
            temp11.append(angle[h,label[h]])
            temp22.append(sum(abs(torch.cat((angle[h,:label[h]], angle[h,label[h]+1:]), axis = 0)-90)))


          for h in range(label.shape[0]):
            if predicted[h] == label[h]:
              continue
            temp33.append(angle[h,label[h]])
            temp44.append((abs(torch.cat((angle[h,:label[h]], angle[h,label[h]+1:]), axis = 0)-90)))         
            
          cc_.append(cc)
          sum_1 = sum(temp11)
          sum_1_3 = sum(temp33)
          sum_2 = (sum(temp22))
          if (len(temp44) == 0):
            sum_2_4 = 0
          else:
            sum_2_4 = sum(sum(temp44))
          final_ = (((sum_1 + sum_2)/correct) + ((sum_1_3 + sum_2_4)/((label.shape[0] - correct)+0.0000000001)))
          max_1.append(max(temp11))
          max_2.append(max(temp22))
          some_new_1.append(sum_1/correct)
          some_new_2.append(sum_2/correct)
          some_new_3.append(sum_1_3/((label.shape[0] - correct)+0.0000000001))
          some_new_4.append(sum_2_4/((label.shape[0] - correct)+0.0000000001))
          some_new.append(final_)
          some_accuracy.append(correct*100/label.shape[0])

        print("max1:",max(max_1),"max2",max(max_2))
        print("A:",sum(some_new_1)/(batch_idx+1),'B:',sum(some_new_2)/(batch_idx+1),'C:',sum(some_new_3)/(batch_idx+1),'D:',sum(some_new_4)/(batch_idx+1))
        print("Sum:",sum(some_new_1)/(batch_idx+1)+sum(some_new_2)/(batch_idx+1)+sum(some_new_3)/(batch_idx+1)+sum(some_new_4)/(batch_idx+1))
        #print("some_new", sum(some_new)/(batch_idx+1))
        print("Accuracy",sum(some_accuracy)/(batch_idx+1))


test(epoch=1)

