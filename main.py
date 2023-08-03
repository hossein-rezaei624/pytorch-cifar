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

from sympy import Matrix

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
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
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


'''classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')'''

# Model
print('==> Building model..')
net = ResNet18()
#net = ResNet34()
#net = ResNet50()
#net = VGG('VGG19')
#net = MobileNetV2()
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



checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])

###print('\n\nLayer params:')
tempp = 0
#weights_ = torch.zeros((512,10))
#bias_ = torch.zeros((10))
for param in net.parameters():
    tempp +=1
    if (tempp==61): ##for example for VGG19 you should set tempp as 65
      ###print(param)
      ###print("the shapeeeeeee",param.shape)
      weights_ = param
    if (tempp==62): ##for example for VGG19 you should set tempp as 66
      ###print(param)
      ###print("the shapeeeeeee",param.shape)
      bias_ = param
#print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",net.state_dict())
print("number of layers of the model",tempp)




def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    count = 0


    with torch.no_grad():        
        counter = 0
        for batch_idx, (img, label) in enumerate(testloader):
          img, label = img.to(device), label.to(device)
          
          outputs, rep = net(img)
          loss = criterion(outputs, label)
          test_loss = loss.item()
          logits__predicted, predicted = outputs.max(1)
          correct = predicted.eq(label).sum().item()
  

          sss = 0
          for label_ in label:
            target_weight = weights_[label_.item(),:]
            #print("eeeeeeeeeeeeee",label_.item())
            #print("target_weight",target_weight.shape)
            other_weight = torch.cat((weights_[:label_.item(),:], weights_[label_.item()+1:,:]), axis = 0)
            #print("target_weight",target_weight.shape, "other_weight",other_weight.shape)
    
            # Calculate the Null space of the matrix
            M = Matrix(other_weight.cpu())
            M_nullspace = M.nullspace()
            #print("dtype", other_weight.dtype)
            bb = np.array(M_nullspace[0])
            bb = bb.astype("float32")
            cc = torch.tensor(bb).to(device)
    
            #print("......",cc.shape)
            #print("rep shape",rep.shape)
            #print("target_weighttttttttt",target_weight.view(512,1).shape)
            #print("rep.shapeeeee",rep.shape)
            a = rep[0,:].view(1,512)
            #print(weights_.shape,"shapeeee")
            #b = cc
            b = target_weight.view(512,1)
            #final = torch.matmul(a,b)+bias_
            #(torch.cat((angle[h,:label[h]], angle[h,label[h]+1:]), axis = 0))
            #print(b.shape)
    
            inner_product = torch.matmul(a,b)
            a_norm = a.pow(2).sum(dim=1).pow(0.5)
            b_norm = b.pow(2).sum(dim=0).pow(0.5)
            hh = torch.matmul(a_norm.view((a_norm.shape[0],1)),b_norm.view((1,1)))
            cos = inner_product / hh
            angle = (torch.acos(cos)*57.2958)
    
            print("angleeeeeeeee",angle)
            sss = sss+1


test(epoch=1)
