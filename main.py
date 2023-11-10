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



num_samples_random = len(trainset) // 4

# Create a random index array
indices_random = np.random.choice(len(trainset), num_samples_random, replace=False)

# Create a subset of the dataset using the random indices
trainset_subset_random = torch.utils.data.Subset(trainset, indices_random)
trainloader_random = torch.utils.data.DataLoader(trainset_subset_random, batch_size=128, shuffle=True, num_workers=0)



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

  

Carto = torch.zeros((4, len(trainset)))

for epoch in range(start_epoch, start_epoch+4):
    train(epoch)
    test(epoch)
    scheduler.step()


Confidence_mean = Carto.mean(dim=0)
Variability = Carto.std(dim=0)


plt.scatter(Variability, Confidence_mean, s = 2)

plt.xlabel("Variability") 
plt.ylabel("Confidence") 

plt.savefig('scatter_plot.png')


top_n = Variability.shape[0]//4

sorted_indices_Variability = np.argsort(Variability.numpy())

top_indices_Variability = sorted_indices_Variability[-top_n:]

top_indices_Variability = top_indices_Variability[::-1]

top_indices_sorted_Variability = top_indices_Variability


subset_data_Variability = torch.utils.data.Subset(trainset, top_indices_sorted_Variability)
trainloader_Variability = torch.utils.data.DataLoader(subset_data_Variability, batch_size=128, shuffle=True, num_workers=0)


# Initialize dictionaries to hold the first image of each class
first_images_Variability = {}
first_labels_Variability = {}

# Iterate over the subset and save the first image of each class
for i, (image, label) in enumerate(subset_data_Variability):
    label = label.item()  # Assuming label is a tensor, convert to int
    if label not in first_images_Variability:
        first_images_Variability[label] = image
        first_labels_Variability[label] = label
    # Break the loop if we have found the first image for each class
    if len(first_images_Variability) == len(trainset.classes):  # Assuming trainset has a 'classes' attribute
        break

# Make a list of the images and labels
images_Variability = list(first_images_Variability.values())
labels_Variability = list(first_labels_Variability.values())

# Make a grid from these images
grid_Variability = torchvision.utils.make_grid(images_Variability, nrow=10)  # Adjust nrow to the number of classes if needed

# Save the grid of images
torchvision.utils.save_image(grid_Variability, 'grid_image_Variability.png')







sorted_indices_Confidence_mean = np.argsort(Confidence_mean.numpy())

top_indices_Confidence_mean = sorted_indices_Confidence_mean[-top_n:]

top_indices_Confidence_mean = top_indices_Confidence_mean[::-1]

top_indices_sorted_Confidence_mean = top_indices_Confidence_mean


subset_data_Confidence_mean = torch.utils.data.Subset(trainset, top_indices_sorted_Confidence_mean)
trainloader_Confidence_mean = torch.utils.data.DataLoader(subset_data_Confidence_mean, batch_size=128, shuffle=True, num_workers=0)


# Extract the first 10 images
images_Confidence_mean = [subset_data_Confidence_mean[i][0] for i in range(400)]
labels_Confidence_mean = [subset_data_Confidence_mean[i][1] for i in range(400)]

# Make a grid from these images
grid_Confidence_mean = torchvision.utils.make_grid(images_Confidence_mean, nrow=20)  # 5 images per row

torchvision.utils.save_image(grid_Confidence_mean, 'grid_image_Confidence_mean.png')






sorted_indices_Confidence_mean_hard = np.argsort(Confidence_mean.numpy())

top_indices_Confidence_mean_hard = sorted_indices_Confidence_mean_hard[:top_n]

top_indices_sorted_Confidence_mean_hard = top_indices_Confidence_mean_hard


subset_data_Confidence_mean_hard = torch.utils.data.Subset(trainset, top_indices_sorted_Confidence_mean_hard)
trainloader_Confidence_mean_hard = torch.utils.data.DataLoader(subset_data_Confidence_mean_hard, batch_size=128, shuffle=True, num_workers=0)


# Extract the first 10 images
images_Confidence_mean_hard = [subset_data_Confidence_mean_hard[i][0] for i in range(400)]
labels_Confidence_mean_hard = [subset_data_Confidence_mean_hard[i][1] for i in range(400)]

# Make a grid from these images
grid_Confidence_mean_hard = torchvision.utils.make_grid(images_Confidence_mean_hard, nrow=20)  # 5 images per row

torchvision.utils.save_image(grid_Confidence_mean_hard, 'grid_image_Confidence_mean_hard.png')





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
for epoch in range(start_epoch, start_epoch+100):
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
for epoch in range(start_epoch, start_epoch+100):
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
for epoch in range(start_epoch, start_epoch+100):
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
for epoch in range(start_epoch, start_epoch+100):
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
for epoch in range(start_epoch, start_epoch+100):
    print("Epoch: ", epoch)
    train_Confidence_mean_hard(epoch)
    test_accuracies_Confidence_mean_hard.append(test_Confidence_mean_hard(epoch))
    scheduler_Confidence_mean_hard.step()






# Clear previous axes and figure
plt.cla()  # Clear the current axes
plt.clf()  # Clear the current figure

epochs = range(start_epoch, start_epoch + 100)

# Plotting
plt.plot(epochs, test_accuracies_all, label='Complete Dataset')
plt.plot(epochs, test_accuracies_random, label='Random Subset (25%)')
plt.plot(epochs, test_accuracies_Variability, label='Most Variable (25%)')
plt.plot(epochs, test_accuracies_Confidence_mean, label='Simplest (25%)')
plt.plot(epochs, test_accuracies_Confidence_mean_hard, label='Most Complex (25%)')


plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Test Accuracy', fontsize=14)
plt.legend(fontsize=12)

plt.ylim(0, 100)

plt.xticks(range(start_epoch, start_epoch + 100, 10), fontsize=12)
plt.yticks(fontsize=12)

plt.savefig("results.png")
