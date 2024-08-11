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
import random


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset_test = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_test)

trainloader_test = torch.utils.data.DataLoader(
    trainset_test, batch_size=128, shuffle=False, num_workers=2, worker_init_fn=lambda worker_id: set_seed(0))


testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=lambda worker_id: set_seed(0))


# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

criterion = nn.CrossEntropyLoss()

checkpoint = torch.load('/home/rezaei/pytorch-cifar/checkpoint/3/ckpt199.pth')
net.load_state_dict(checkpoint['net'])

# Accessing the last fully connected layer correctly
last_fc = net.module.linear if hasattr(net, 'module') else net.linear
W = last_fc.weight.data

def cosine_similarity(rep, W):
    # Normalize the representation and the weights
    W_norm = F.normalize(W, p=2, dim=1)
    rep_norm = F.normalize(rep, p=2, dim=1)
    return torch.mm(rep_norm, W_norm.t())

def norm_of_projection_all(rep, W):
    W_norm = F.normalize(W, p=2, dim=1)
    dot_products = torch.mm(rep, W_norm.t())
    projections = dot_products.unsqueeze(2) * W_norm.unsqueeze(0)
    norms = torch.norm(projections, dim=2)
    return norms


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    cos_sim_batch_target = []
    cos_sim_batch_non_target = []
    proj_batch_target = []
    proj_batch_non_target = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            representations, logits = net(inputs)
            
            cos_sim = cosine_similarity(representations, W)
            projection_norms = norm_of_projection_all(representations, W)

            cos_sim_sample_target = []
            cos_sim_sample_non_target = []
            proj_sample_target = []
            proj_sample_non_target = []
            
            for i in range(inputs.size(0)):
                target_class = targets[i].item()
                
                target_cos_sim = cos_sim[i, target_class]
                target_proj = projection_norms[i, target_class]

                non_target_cos_sim = torch.cat((cos_sim[i, :target_class], cos_sim[i, target_class+1:]))   
                mean_cos_sim_nontarget = torch.abs(non_target_cos_sim).mean()

                non_target_proj = torch.cat((projection_norms[i, :target_class], projection_norms[i, target_class+1:]))                
                mean_proj_nontarget = torch.abs(non_target_proj).mean()
    
                cos_sim_sample_target.append(target_cos_sim.item())
                cos_sim_sample_non_target.append(mean_cos_sim_nontarget.item())
                proj_sample_target.append(target_proj.item())
                proj_sample_non_target.append(mean_proj_nontarget.item())

            
            cos_sim_batch_target.append(np.mean(cos_sim_sample_target))
            cos_sim_batch_non_target.append(np.mean(cos_sim_sample_non_target))
            proj_batch_target.append(np.mean(proj_sample_target))
            proj_batch_non_target.append(np.mean(proj_sample_non_target))

            
            loss = criterion(logits, targets)
            test_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
          
        print("\nTest Accuracy:", 100.*correct/total)
        
        print("cos_sim_batch_target", np.mean(cos_sim_batch_target))
        print("cos_sim_batch_non_target", np.mean(cos_sim_batch_non_target))
        print("proj_batch_target", np.mean(proj_batch_target))
        print("proj_batch_non_target", np.mean(proj_batch_non_target))

def test_train(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader_test):
            inputs, targets = inputs.to(device), targets.to(device)
            representations, logits = net(inputs)

            loss = criterion(logits, targets)
            test_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print("\nTrain Accuracy on eval mode", 100.*correct/total)


test(1)
test_train(1)
