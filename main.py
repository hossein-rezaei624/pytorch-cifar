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
    te = 128
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, rep_1, weights_1, bias_1 = net(inputs)
        #print("the labellll:", targets, targets.shape, targets[0])
        ###print("outputs.shape", outputs.shape,"rep_1.shape", rep_1.shape, "weights_1.shape",weights_1.shape, "bias_1.shape",bias_1.shape, "inputs.shape",inputs.shape)
        temp_1 = []
        temp_1_1 = []
        sum_ = []
        ##print("representation:",rep_1.shape)
        #print("batch_idx",batch_idx)
        if (batch_idx == 390):
          te = 80
      #for j in range(te):
        #angle = []  
        #for i in range(10):

        a = rep_1
        #print("batch_idx",batch_idx)
        b = weights_1.transpose(0,1)
        #print("aaaaa shape",a.shape)
        #print("bbbbbbbbb shape",b.shape)
        final = torch.matmul(a,b)+bias_1
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
        #print("angle",angle, angle.shape)

        #print("The angle with the weights of the class",i," is:",angle*57.2958)
        #print("the angle isssss:", angle, "\n the label",angle[targets[0]],"ddd",targets[0])

        for h in range(te):
          temp_1.append(angle[h,targets[h]])
          temp_1_1.append(torch.cat((angle[h,:targets[h]], angle[h,targets[h]+1:]), axis = 0))
        #print("temp_1",len(temp_1),temp_1[0])
        #print("temp_1_1",len(temp_1_1),temp_1_1[0])
        ###del angle[targets[j]]
        ##print("afterrr",angle)
        #sum_ = (sum(temp_1_1))
        ##print("sum_",sum_)
        #print("ddd",0.1*temp_1,"aaa",(1000/sum_))
        #///////////////////////
        temp_2 = sum(temp_1)
        #print("temp_2",(temp_2))
        sum_1 = sum(sum(temp_1_1))
        #print("sum_1",(sum_1))
        #criterion(outputs, targets)
        loss = criterion(outputs, targets) + torch.abs(0.01*(sum_1 - (te*9*90))) + (0.0001*temp_2)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    print("jjjjjjjjjjjj",0.0001*temp_2,"hhhhhhhhh",torch.abs(0.01*(sum_1 - (te*9*90))))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, __, ___, ____ = net(inputs)
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


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
