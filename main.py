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
    trainset, batch_size=128, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
print("lennnnnnnnnnn of the testloader",len(testloader))

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
weights_ = torch.zeros((512,10))
bias_ = torch.zeros((10))
for param in net.parameters():
    tempp +=1
    if (tempp==61):
      ###print(param)
      ###print("the shapeeeeeee",param.shape)
      weights_ = param
    if (tempp==62):
      ###print(param)
      ###print("the shapeeeeeee",param.shape)
      bias_ = param
#print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",net.state_dict())
#print("temppppppppp",tempp)
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
    true_list = []
    false_list = []
    with torch.no_grad():
        '''#img, label = testset[0]
        
        #for i in range(324):
          #img, label = next(iter(testloader))
        img, label = next(iter(testloader))
        print("img shape:",img.shape,img[0].shape,"label",label)
        img, label = img[10].view((1,3,32,32)), label[10].view((1))
        print("img shapeeeeeee:",img.shape,"label",label)
        img, label = img.to(device), label.to(device)'''
        
        counter = 0
        for batch_idx, (img, label) in enumerate(testloader):
          
          #img = torch.tensor(random_noise(img, mode='salt', amount=0.05))
          #img = img + torch.randn(img.size()) * 0.3 + 0.0
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
          ####print("Loss:",test_loss,"Accuracy:",correct*100/100)
          '''print("manual",torch.matmul(rep,weights_.transpose(0,1))+bias_)
          print("manual shape",(torch.matmul(rep,weights_.transpose(0,1))+bias_).shape)
          mm_ = torch.nn.Softmax(dim=-1)
          output__ = mm_(torch.matmul(rep,weights_.transpose(0,1))+bias_)
          print(output__)'''

          '''for i in range(10):

            a = rep[0,:]
            b = weights_[i,:]
            #print("aaaaa shape",a.shape)
            #print("bbbbbbbbb shape",b.shape)
            final = torch.matmul(a,b)+bias_[i]
            print("final",i,":",final)

            inner_product = (a * b).sum(dim=0)
            #print(inner_product)
            a_norm = a.pow(2).sum(dim=0).pow(0.5)
            #print(a_norm)
            b_norm = b.pow(2).sum(dim=0).pow(0.5)
            cos = inner_product / (a_norm * b_norm)
            #print(cos)
            angle = torch.acos(cos)

            print("The angle with the weights of the class",i," is:",angle*57.2958)'''


          a = rep
          #print("batch_idx",batch_idx)
          b = weights_.transpose(0,1)
          #print("aaaaa shape",a.shape)
          #print("bbbbbbbbb shape",b.shape)
          final = torch.matmul(a,b)+bias_
          #print("final:",final)

          inner_product = torch.matmul(a,b)
          #print('inner_product',inner_product.shape)
          a_norm = a.pow(2).sum(dim=1).pow(0.5)
          #print('a_norm',a_norm.shape)
          b_norm = b.pow(2).sum(dim=0).pow(0.5)
          #print('b_norm',b_norm.shape)
          hh = torch.matmul(a_norm.view((a_norm.shape[0],1)),b_norm.view((1,10)))
          #print("torch.matmul(a_norm,b_norm) shape",hh.shape)
          cos = inner_product / hh
          #print('cos',cos,cos.shape)
          angle = (torch.acos(cos)*57.2958)

          true_ = 0
          false_ = 0
          for h in range(label.shape[0]):
            aa__ = min(angle[h,:]).item()
            temp11.append(aa__)
            bb__ = (angle[h,:] == aa__).nonzero(as_tuple=True)[0].item()
            cc__ = (abs(torch.cat((angle[h,:bb__], angle[h,bb__+1:]), axis = 0)-90))
            hh_ = aa__ + sum(cc__)
            if (hh_ < 142):
              true_ +=1
            else:
              false_ +=1
              
          true_list.append(true_*100/label.shape[0])
          false_list.append(false_*100/label.shape[0])

        print("Accuracy",sum(true_list)/(batch_idx+1))
          

test(epoch=1)
