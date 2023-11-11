import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse

from models import ResNet18
from utils import progress_bar

import numpy as np
import matplotlib.pyplot as plt
import torchvision

import random

from torch.utils.data import Dataset
import pickle

from collections import defaultdict
from torch.utils.data import Subset


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
  torch.cuda.manual_seed(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch




def distribute_samples(probabilities, M):

    # Normalize the probabilities
    total_probability = sum(probabilities.values())
    normalized_probabilities = {k: v / total_probability for k, v in probabilities.items()}

    # Calculate the number of samples for each class
    samples = {k: round(v * M) for k, v in normalized_probabilities.items()}
    
    # Check if there's any discrepancy due to rounding and correct it
    discrepancy = M - sum(samples.values())
    
    for key in samples:
        if discrepancy == 0:
            break
        if discrepancy > 0:
            samples[key] += 1
            discrepancy -= 1
        elif discrepancy < 0 and samples[key] > 0:
            samples[key] -= 1
            discrepancy += 1

    return samples


def distribute_excess(lst):
    # Calculate the total excess value
    total_excess = sum(val - 500 for val in lst if val > 500)

    # Number of elements that are not greater than 500
    recipients = [i for i, val in enumerate(lst) if val < 500]

    num_recipients = len(recipients)

    # Calculate the average share and remainder
    avg_share, remainder = divmod(total_excess, num_recipients)

    lst = [val if val <= 500 else 500 for val in lst]
    
    # Distribute the average share
    for idx in recipients:
        lst[idx] += avg_share
    
    # Distribute the remainder
    for idx in recipients[:remainder]:
        lst[idx] += 1
    
    # Cap values greater than 500
    for i, val in enumerate(lst):
        if val > 500:
            return distribute_excess(lst)
            break

    return lst






# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=0)



trainloader_all = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)



num_samples_random = len(trainset) // 10

# Create a random index array
indices_random = np.random.choice(len(trainset), num_samples_random, replace=False)

# Create a subset of the dataset using the random indices
trainset_subset_random = torch.utils.data.Subset(trainset, indices_random)
trainloader_random = torch.utils.data.DataLoader(trainset_subset_random, batch_size=64, shuffle=True, num_workers=0)



# Model
print('==> Building model..')


net = ResNet18(10)
net_all = ResNet18(10)
net_random = ResNet18(10)
net_Variability = ResNet18(10)
net_Confidence_mean = ResNet18(10)
net_Confidence_mean_hard = ResNet18(10)


net = net.to(device)
net_all = net_all.to(device)
net_random = net_random.to(device)
net_Variability = net_Variability.to(device)
net_Confidence_mean = net_Confidence_mean.to(device)
net_Confidence_mean_hard = net_Confidence_mean_hard.to(device)


criterion = nn.CrossEntropyLoss()
criterion_all = nn.CrossEntropyLoss()
criterion_random = nn.CrossEntropyLoss()
criterion_Variability = nn.CrossEntropyLoss()
criterion_Confidence_mean = nn.CrossEntropyLoss()
criterion_Confidence_mean_hard = nn.CrossEntropyLoss()



optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
optimizer_all = optim.SGD(net_all.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
optimizer_random = optim.SGD(net_random.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
optimizer_Variability = optim.SGD(net_Variability.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
optimizer_Confidence_mean = optim.SGD(net_Confidence_mean.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
optimizer_Confidence_mean_hard = optim.SGD(net_Confidence_mean_hard.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)



scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler_all = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_all, T_max=200)
scheduler_random = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_random, T_max=200)
scheduler_Variability = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_Variability, T_max=200)
scheduler_Confidence_mean = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_Confidence_mean, T_max=200)
scheduler_Confidence_mean_hard = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_Confidence_mean_hard, T_max=200)

unique_classes = set()
for _, labels, indices_1 in trainloader:
    unique_classes.update(labels.numpy())



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    confidence_epoch = []
    for batch_idx, (inputs, targets, indices_1) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, soft_ = net(inputs)
        confidence_batch = []
        for i in range(targets.shape[0]):
          confidence_batch.append(soft_[i,targets[i]].item())

          # Update the dictionary with the confidence score for the current class for the current epoch
          confidence_by_class[targets[i].item()][epoch].append(soft_[i, targets[i]].item())
      
        # ... [Rest of the batch processing code]
        conf_tensor = torch.tensor(confidence_batch)
        Carto[epoch, indices_1] = conf_tensor  # Place confidences in the right location using indices

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
  

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indices_1) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, __ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

  


confidence_by_class = {class_id: {epoch: [] for epoch in range(4)} for class_id, __ in enumerate(unique_classes)}

Carto = torch.zeros((4, len(trainset)))
for epoch in range(start_epoch, start_epoch+4):
    train(epoch)
    test(epoch)
    scheduler.step()


mean_by_class = {class_id: {epoch: torch.mean(torch.tensor(confidences[epoch])) for epoch in confidences} for class_id, confidences in confidence_by_class.items()}
std_of_means_by_class = {class_id: torch.std(torch.tensor([mean_by_class[class_id][epoch] for epoch in range(4)])) for class_id, __ in enumerate(unique_classes)}
mean_of_means_by_class = {class_id: torch.mean(torch.tensor([mean_by_class[class_id][epoch] for epoch in range(4)])) for class_id, __ in enumerate(unique_classes)}



top_n = len(trainset) // 10


updated_std_of_means_by_class = {k: v.item() for k, v in std_of_means_by_class.items()}

print("updated_std_of_means_by_class", updated_std_of_means_by_class)

dist = distribute_samples(updated_std_of_means_by_class, top_n)


num_per_class = top_n//len(trainset.classes)
counter_class = [0 for _ in range(len(trainset.classes))]

condition = [value for k, value in dist.items()]

check_bound = len(trainset)/len(trainset.classes)

for i in range(len(condition)):
    if condition[i] > check_bound:
        condition = distribute_excess(condition)
        break


class_indices = defaultdict(list)
for idx, (_, label, __) in enumerate(trainset):
    class_indices[label.item()].append(idx)

selected_indices = []

for class_id, num_samples in enumerate(condition):
    class_samples = class_indices[class_id]  # get indices for the class
    selected_for_class = random.sample(class_samples, num_samples)
    selected_indices.extend(selected_for_class)

selected_dataset_Variability = Subset(trainset, selected_indices)
trainloader_Variability = torch.utils.data.DataLoader(selected_dataset_Variability, batch_size=64, shuffle=True, num_workers=0)





updated_mean_of_means_by_class = {k: v.item() for k, v in mean_of_means_by_class.items()}

print("updated_mean_of_means_by_class", updated_mean_of_means_by_class)

dist = distribute_samples(updated_mean_of_means_by_class, top_n)


num_per_class = top_n//len(trainset.classes)
counter_class = [0 for _ in range(len(trainset.classes))]

condition = [value for k, value in dist.items()]

check_bound = len(trainset)/len(trainset.classes)

for i in range(len(condition)):
    if condition[i] > check_bound:
        condition = distribute_excess(condition)
        break


class_indices = defaultdict(list)
for idx, (_, label, __) in enumerate(trainset):
    class_indices[label.item()].append(idx)

selected_indices = []

for class_id, num_samples in enumerate(condition):
    class_samples = class_indices[class_id]  # get indices for the class
    selected_for_class = random.sample(class_samples, num_samples)
    selected_indices.extend(selected_for_class)

selected_dataset_Confidence_mean = Subset(trainset, selected_indices)
trainloader_Confidence_mean = torch.utils.data.DataLoader(selected_dataset_Confidence_mean, batch_size=64, shuffle=True, num_workers=0)







updated_mean_of_means_by_class = {k: 1 - v.item() for k, v in mean_of_means_by_class.items()}

dist = distribute_samples(updated_mean_of_means_by_class, top_n)

num_per_class = top_n//len(trainset.classes)
counter_class = [0 for _ in range(len(trainset.classes))]

condition = [value for k, value in dist.items()]

check_bound = len(trainset)/len(trainset.classes)

for i in range(len(condition)):
    if condition[i] > check_bound:
        condition = distribute_excess(condition)
        break


class_indices = defaultdict(list)
for idx, (_, label, __) in enumerate(trainset):
    class_indices[label.item()].append(idx)

selected_indices = []

for class_id, num_samples in enumerate(condition):
    class_samples = class_indices[class_id]  # get indices for the class
    selected_for_class = random.sample(class_samples, num_samples)
    selected_indices.extend(selected_for_class)

selected_dataset_Confidence_mean_hard = Subset(trainset, selected_indices)
trainloader_Confidence_mean_hard = torch.utils.data.DataLoader(selected_dataset_Confidence_mean_hard, batch_size=64, shuffle=True, num_workers=0)







def train_all(epoch):
  net_all.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets, indices_1) in enumerate(trainloader_all):
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer_all.zero_grad()
      outputs, soft_ = net_all(inputs)

      loss = criterion_all(outputs, targets)
      loss.backward()
      optimizer_all.step()

      train_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      progress_bar(batch_idx, len(trainloader_all), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    

def test_all(epoch):
    net_all.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indices_1) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, __ = net_all(inputs)
            loss = criterion_all(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


          
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print("\n")
    test_accuracy = 100. * correct / total
    return test_accuracy
        
print("\n")
print("Trainning with all...")
print("\n")
test_accuracies_all = []
for epoch in range(start_epoch, start_epoch+21):
    print("Epoch: ", epoch)
    train_all(epoch)
    test_accuracies_all.append(test_all(epoch))
    scheduler_all.step()






def train_random(epoch):
  net_random.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets, indices_1) in enumerate(trainloader_random):
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer_random.zero_grad()
      outputs, soft_ = net_random(inputs)

      loss = criterion_random(outputs, targets)
      loss.backward()
      optimizer_random.step()

      train_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      progress_bar(batch_idx, len(trainloader_random), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    

def test_random(epoch):
    net_random.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indices_1) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, __ = net_random(inputs)
            loss = criterion_random(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


          
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print("\n")
    test_accuracy = 100. * correct / total
    return test_accuracy
        
print("\n")
print("Trainning with random...")
print("\n")
test_accuracies_random = []
for epoch in range(start_epoch, start_epoch+21):
    print("Epoch: ", epoch)
    train_random(epoch)
    test_accuracies_random.append(test_random(epoch))
    scheduler_random.step()






def train_Variability(epoch):
  net_Variability.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets, indices_1) in enumerate(trainloader_Variability):
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer_Variability.zero_grad()
      outputs, soft_ = net_Variability(inputs)

      loss = criterion_Variability(outputs, targets)
      loss.backward()
      optimizer_Variability.step()

      train_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      progress_bar(batch_idx, len(trainloader_Variability), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    

def test_Variability(epoch):
    net_Variability.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indices_1) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, __ = net_Variability(inputs)
            loss = criterion_Variability(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


          
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print("\n")
    test_accuracy = 100. * correct / total
    return test_accuracy
        
print("\n")
print("Trainning with Variability...")
print("\n")
test_accuracies_Variability = []
for epoch in range(start_epoch, start_epoch+21):
    print("Epoch: ", epoch)
    train_Variability(epoch)
    test_accuracies_Variability.append(test_Variability(epoch))
    scheduler_Variability.step()










def train_Confidence_mean(epoch):
  net_Confidence_mean.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets, indices_1) in enumerate(trainloader_Confidence_mean):
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer_Confidence_mean.zero_grad()
      outputs, soft_ = net_Confidence_mean(inputs)

      loss = criterion_Confidence_mean(outputs, targets)
      loss.backward()
      optimizer_Confidence_mean.step()

      train_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      progress_bar(batch_idx, len(trainloader_Confidence_mean), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    

def test_Confidence_mean(epoch):
    net_Confidence_mean.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indices_1) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, __ = net_Confidence_mean(inputs)
            loss = criterion_Confidence_mean(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


          
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print("\n")
    test_accuracy = 100. * correct / total
    return test_accuracy
        
print("\n")
print("Trainning with Confidence_mean...")
print("\n")
test_accuracies_Confidence_mean = []
for epoch in range(start_epoch, start_epoch+21):
    print("Epoch: ", epoch)
    train_Confidence_mean(epoch)
    test_accuracies_Confidence_mean.append(test_Confidence_mean(epoch))
    scheduler_Confidence_mean.step()











def train_Confidence_mean_hard(epoch):
  net_Confidence_mean_hard.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets, indices_1) in enumerate(trainloader_Confidence_mean_hard):
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer_Confidence_mean_hard.zero_grad()
      outputs, soft_ = net_Confidence_mean_hard(inputs)

      loss = criterion_Confidence_mean_hard(outputs, targets)
      loss.backward()
      optimizer_Confidence_mean_hard.step()

      train_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      progress_bar(batch_idx, len(trainloader_Confidence_mean_hard), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    

def test_Confidence_mean_hard(epoch):
    net_Confidence_mean_hard.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indices_1) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, __ = net_Confidence_mean_hard(inputs)
            loss = criterion_Confidence_mean_hard(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


          
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print("\n")
    test_accuracy = 100. * correct / total
    return test_accuracy
        
print("\n")
print("Trainning with Confidence_mean_hard...")
print("\n")
test_accuracies_Confidence_mean_hard = []
for epoch in range(start_epoch, start_epoch+21):
    print("Epoch: ", epoch)
    train_Confidence_mean_hard(epoch)
    test_accuracies_Confidence_mean_hard.append(test_Confidence_mean_hard(epoch))
    scheduler_Confidence_mean_hard.step()




# Clear previous axes and figure
plt.cla()  # Clear the current axes
plt.clf()  # Clear the current figure

epochs = range(start_epoch, start_epoch + 21)

# Plotting
plt.plot(epochs, test_accuracies_all)
plt.plot(epochs, test_accuracies_random)
plt.plot(epochs, test_accuracies_Variability)
plt.plot(epochs, test_accuracies_Confidence_mean)
plt.plot(epochs, test_accuracies_Confidence_mean_hard)

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Test Accuracy', fontsize=14)

plt.ylim(0, 100)

plt.xticks(range(start_epoch, start_epoch + 21, 2), fontsize=12)
plt.yticks(fontsize=12)

plt.savefig("resultsClass.png")
