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


parser = argparse.ArgumentParser(description='Starting..')
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

checkpoint = torch.load('/home/rezaei/thesis/pytorch-cifar/checkpoint/cifar10/resnet18_1/ckpt199.pth')
net.load_state_dict(checkpoint['net'])

# Accessing the last fully connected layer correctly
last_fc = net.module.linear if hasattr(net, 'module') else net.linear
W = last_fc.weight.data

def calc_projection_matrices(W):
    """Calculate column and null space projection matrices for a given weight matrix W."""
    WWt_pinv = torch.pinverse(W @ W.T)
    proj_column_space = W.T @ WWt_pinv @ W
    proj_null_space = torch.eye(W.T.shape[0], device=W.device) - proj_column_space
    return proj_null_space

def precompute_projections(W):
    projections = {}
    for i in range(W.shape[0]):  # Assume W is (10, 512)
        # Isolate weight for non-target classes
        non_target_weights = torch.cat([W[:i], W[i+1:]], dim=0)  # Shape (9, 512)

        # Calculate projection matrices
        proj_null_space_nontarget = calc_projection_matrices(non_target_weights)

        # Store in dictionary
        projections[i] = {
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

    angle_target_list = []
    angle_null_non_target_list = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            representations, logits = net(inputs)

            # Apply precomputed projections
            for i in range(representations.size(0)):  # Loop over the batch
                target_class = targets[i].item()
                proj_info = projections[target_class]
    
                output_vector = representations[i]

                # Get target weight vector and reshape to (512,)
                target_weight = W[target_class].squeeze()  # Shape (512,)

                # Compute cosine similarity between output_vector and target_weight
                cos_theta_target = torch.dot(output_vector, target_weight) / (torch.norm(output_vector) * torch.norm(target_weight))
                # Clamp cos_theta_target to [-1,1] to avoid NaNs
                cos_theta_target = torch.clamp(cos_theta_target, -1.0, 1.0)
                # Compute angle in degrees
                theta_target = torch.acos(cos_theta_target) * (180 / np.pi)
                # Append to list
                angle_target_list.append(theta_target.item())

                # Compute null space projection of non-target classes
                null_space_repr_non_target = proj_info['null_nontarget'] @ output_vector  # Shape (512,1)
                null_space_repr_non_target_squeezed = null_space_repr_non_target.squeeze()  # Shape (512,)

                # Compute cosine similarity between output_vector and null_space_repr_non_target
                norm_null_space_repr_non_target = torch.norm(null_space_repr_non_target_squeezed)
                if norm_null_space_repr_non_target > 0:
                    cos_theta_null_non_target = torch.dot(output_vector, null_space_repr_non_target_squeezed) / (torch.norm(output_vector) * norm_null_space_repr_non_target)
                    cos_theta_null_non_target = torch.clamp(cos_theta_null_non_target, -1.0, 1.0)
                    theta_null_non_target = torch.acos(cos_theta_null_non_target) * (180 / np.pi)
                else:
                    # If the null space projection is zero vector, set angle to 90 degrees
                    theta_null_non_target = 90.0

                # Append to list
                angle_null_non_target_list.append(theta_null_non_target.item())
            
            loss = criterion(logits, targets)
            test_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
          
        print("\nTest Accuracy:", 100.*correct/total)

        angle_target_mean = np.mean(angle_target_list)
        angle_null_non_target_mean = np.mean(angle_null_non_target_list)
        
        print("Average Angles:")
        print("Angle between representation and target class weight (degrees):", angle_target_mean)
        print("Angle between representation and null space of non-target classes (degrees):", angle_null_non_target_mean - 90)


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

        print("\nTrain Accuracy on eval mode", 100.*correct/total, "\n")


test(1)
test_train(1)
