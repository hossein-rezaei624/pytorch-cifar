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

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


'''classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')'''

# Model
print('==> Building model..')
#net = ResNet18()
net = DenseNet121()
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
    if (tempp==361): ##for example for VGG19 you should set tempp as 65
      ###print(param)
      ###print("the shapeeeeeee",param.shape)
      weights_ = param
    if (tempp==362): ##for example for VGG19 you should set tempp as 66
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
      
        list_null = []
        for i in range(100):
          
          other_weight = torch.cat((weights_[:i,:], weights_[i+1:,:]), axis = 0)
          # Calculate the Null space of the matrix
          M = Matrix(other_weight.cpu())
          M_nullspace = M.nullspace()
          #print("dtype", other_weight.dtype)
          bb = np.array(M_nullspace[0])
          bb = bb.astype("float32")
          #cc = torch.tensor(bb).to(device)
          list_null.append(bb)
          print("iiiiii",i)

        torch_null = torch.squeeze(torch.tensor(np.array(list_null)).to(device))
        print("torch_null", torch_null.shape)
        list1_1 = []
        suum1 = []
        suum2 = []
        for batch_idx, (img, label) in enumerate(testloader):
          img, label = img.to(device), label.to(device)
          
          outputs, rep = net(img)
          loss = criterion(outputs, label)
          test_loss = loss.item()
          logits__predicted, predicted = outputs.max(1)
          correct = predicted.eq(label).sum().item()
  
          a = rep
          b = weights_.transpose(0,1)
          #print('rep shape',a.shape)
          #print("null shape",b.shape)
  
          inner_product = torch.matmul(a,b)
          a_norm = a.pow(2).sum(dim=1).pow(0.5)
          b_norm = b.pow(2).sum(dim=0).pow(0.5)
          hh = torch.matmul(a_norm.view((a_norm.shape[0],1)),b_norm.view((1,10)))
          cos = inner_product / hh
          angle_target = (torch.acos(cos)*57.2958)
          #print("angle_target shape", angle_target.shape)


          a = rep
          b = torch_null.transpose(0,1)
          #print('rep shape',a.shape)
          #print("null shape",b.shape)
  
          inner_product = torch.matmul(a,b)
          a_norm = a.pow(2).sum(dim=1).pow(0.5)
          b_norm = b.pow(2).sum(dim=0).pow(0.5)
          hh = torch.matmul(a_norm.view((a_norm.shape[0],1)),b_norm.view((1,10)))
          cos = inner_product / hh
          angle_null = (torch.acos(cos)*57.2958)

          #print("angle_null shape", angle_null.shape)

          count1 = 0
          sum_1_1 = []
          sum_2_2 = []
          for label_1 in label:
            sum_1_1.append(angle_target[count1, label_1.item()])
            sum_2_2.append(angle_null[count1, label_1.item()])
            count1 = count1 + 1

          #print("sum_1_1",sum_1_1,len(sum_1_1))
          #print("sum_2_2",sum_2_2,len(sum_2_2))
          suum1.append(sum(sum_1_1)/len(label))
          suum2.append(sum(sum_2_2)/len(label))

        print("final target angle is:", sum(suum1)/(batch_idx+1))
        print("final null angle is:", sum(suum2)/(batch_idx+1))
            
###

test(epoch=1)
