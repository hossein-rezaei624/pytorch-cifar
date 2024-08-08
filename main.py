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
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset_test = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_test)

trainloader_test = torch.utils.data.DataLoader(
    trainset_test, batch_size=128, shuffle=False, num_workers=2, worker_init_fn=lambda worker_id: set_seed(0))


testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=lambda worker_id: set_seed(0))

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

criterion = nn.CrossEntropyLoss()

checkpoint = torch.load('/home/rezaei/pytorch-cifar/checkpoint/resnet18_2/ckpt199.pth')
net.load_state_dict(checkpoint['net'])

# Accessing the last fully connected layer correctly
last_fc = net.module.linear if hasattr(net, 'module') else net.linear
W = last_fc.weight.data

def calc_projection_matrices(W):
    """Calculate column and null space projection matrices for a given weight matrix W."""
    WWt_pinv = torch.pinverse(W @ W.T)
    proj_column_space = W.T @ WWt_pinv @ W
    proj_null_space = torch.eye(W.T.shape[0], device=W.device) - proj_column_space
    return proj_column_space, proj_null_space

def precompute_projections(W):
    """Precompute projections for all classes."""
    projections = {}
    for i in range(W.shape[0]):  # Assume W is (10, 512)
        # Isolate weight for target and non-target classes
        target_weight = W[i].unsqueeze(0)  # Shape (1, 512)
        non_target_weights = torch.cat([W[:i], W[i+1:]], dim=0)  # Shape (9, 512)

        # Calculate projection matrices
        proj_column_space_target, proj_null_space_target = calc_projection_matrices(target_weight)
        proj_column_space_nontarget, proj_null_space_nontarget = calc_projection_matrices(non_target_weights)

        # Store in dictionary
        projections[i] = {
            'column_target': proj_column_space_target,
            'null_target': proj_null_space_target,
            'column_nontarget': proj_column_space_nontarget,
            'null_nontarget': proj_null_space_nontarget
        }
    return projections

# Precompute the projection matrices
projections = precompute_projections(W)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    col_target_list = []
    col_non_target_list = []
    null_target_list = []
    null_non_target_list = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            representations, logits = net(inputs)

            # Apply precomputed projections
            for i in range(representations.size(0)):  # Loop over the batch
                target_class = targets[i].item()
                proj_info = projections[target_class]
    
                # Reshape outputs[i] to (512, 1) for proper matrix multiplication
                output_vector = representations[i].unsqueeze(1)  # Now shape (512, 1)
                
                col_space_repr_target = proj_info['column_target'] @ output_vector
                col_space_repr_non_target = proj_info['column_nontarget'] @ output_vector

                null_space_repr_target = proj_info['null_target'] @ output_vector
                null_space_repr_non_target = proj_info['null_nontarget'] @ output_vector

                target_weight = W[target_class].unsqueeze(0)  # Shape (1, 512)
                non_target_weights = torch.cat([W[:target_class], W[target_class+1:]], dim=0)  # Shape (9, 512)
                
    
                col_space_repr_target_norm = torch.norm(col_space_repr_target, dim=0)/(torch.norm(target_weight, p='fro'))
                col_space_repr_non_target_norm = torch.norm(col_space_repr_non_target, dim=0)/(torch.norm(non_target_weights, p='fro'))
                null_space_repr_target_norm = torch.norm(null_space_repr_target, dim=0)/(torch.norm(target_weight, p='fro'))
                null_space_repr_non_target_norm = torch.norm(null_space_repr_non_target, dim=0)/(torch.norm(non_target_weights, p='fro'))

                col_target_list.append(col_space_repr_target_norm.item())
                col_non_target_list.append(col_space_repr_non_target_norm.item())
                null_target_list.append(null_space_repr_target_norm.item())
                null_non_target_list.append(null_space_repr_non_target_norm.item())
                
            
            loss = criterion(logits, targets)
            test_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
          
        print("\nTest Accuracy:", 100.*correct/total)

        col_target_mean = np.mean(col_target_list)
        col_non_target_mean = np.mean(col_non_target_list)
        null_target_mean = np.mean(null_target_list)
        null_non_target_mean = np.mean(null_non_target_list)
        
        print("Sample Projection Outputs for Test:")
        print("col_target_mean:", col_target_mean)
        print("col_non_target_mean:", col_non_target_mean)
        print("null_target_mean:", null_target_mean)
        print("null_non_target_mean:", null_non_target_mean)


def test_train(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    col_target_list = []
    col_non_target_list = []
    null_target_list = []
    null_non_target_list = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader_test):
            inputs, targets = inputs.to(device), targets.to(device)
            representations, logits = net(inputs)

            # Apply precomputed projections
            for i in range(representations.size(0)):  # Loop over the batch
                target_class = targets[i].item()
                proj_info = projections[target_class]
    
                # Reshape outputs[i] to (512, 1) for proper matrix multiplication
                output_vector = representations[i].unsqueeze(1)  # Now shape (512, 1)
                
                col_space_repr_target = proj_info['column_target'] @ output_vector
                col_space_repr_non_target = proj_info['column_nontarget'] @ output_vector

                null_space_repr_target = proj_info['null_target'] @ output_vector
                null_space_repr_non_target = proj_info['null_nontarget'] @ output_vector
                
                target_weight = W[target_class].unsqueeze(0)  # Shape (1, 512)
                non_target_weights = torch.cat([W[:target_class], W[target_class+1:]], dim=0)  # Shape (9, 512)
                
                col_space_repr_target_norm = torch.norm(col_space_repr_target, dim=0)/(torch.norm(target_weight, p='fro'))
                col_space_repr_non_target_norm = torch.norm(col_space_repr_non_target, dim=0)/(torch.norm(non_target_weights, p='fro'))
                null_space_repr_target_norm = torch.norm(null_space_repr_target, dim=0)/(torch.norm(target_weight, p='fro'))
                null_space_repr_non_target_norm = torch.norm(null_space_repr_non_target, dim=0)/(torch.norm(non_target_weights, p='fro'))

                col_target_list.append(col_space_repr_target_norm.item())
                col_non_target_list.append(col_space_repr_non_target_norm.item())
                null_target_list.append(null_space_repr_target_norm.item())
                null_non_target_list.append(null_space_repr_non_target_norm.item())
            
            loss = criterion(logits, targets)
            test_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print("\nTrain Accuracy on eval mode", 100.*correct/total)

        col_target_mean = np.mean(col_target_list)
        col_non_target_mean = np.mean(col_non_target_list)
        null_target_mean = np.mean(null_target_list)
        null_non_target_mean = np.mean(null_non_target_list)
        
        print("Sample Projection Outputs for Test:")
        print("col_target_mean:", col_target_mean)
        print("col_non_target_mean:", col_non_target_mean)
        print("null_target_mean:", null_target_mean)
        print("null_non_target_mean:", null_non_target_mean, '\n')


test(1)
test_train(1)
