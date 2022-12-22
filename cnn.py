import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch.optim as optim
import torch.nn.functional as F


#device = torch.device('mps')
train_path = 'fashion_train.npy'
test_path = 'fashion_test.npy'

class CustomDataset(Dataset):
    def __init__(self, npy_file, transform=None):
        self.data = np.load(npy_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target = self.data[idx, -1]
        image = np.reshape(self.data[idx, :-1], (-1, 28))
       
        sample = {'image': image, 'target': target} 

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #print(x.shape)
        #print(F.relu(self.fc1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


batch_size=4
norm_metric = 1/255

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(norm_metric, norm_metric),
     transforms.Grayscale(num_output_channels=1)])
    
train_set = CustomDataset(train_path, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = CustomDataset(test_path, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Shirt')


model = Net()
#model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        #inputs, labels = Variable(data['image'].to(device)), Variable(data['target'].to(device))
        inputs, labels = Variable(data['image']), Variable(data['target'])
        
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)

        #compute loss
        loss = criterion(outputs, labels)

        # backpropagate
        loss.backward()

        # update weights
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

dataiter = iter(test_loader)
images, labels = next(dataiter)

# print images
_, predicted = torch.max(outputs, 1)

#print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
#                              for j in range(4)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        #images, labels = data
        images, labels = Variable(data['image']), Variable(data['target'])
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test data accuracy_ {100 * correct // total} %')


#if __name__ == '__main__':
#    pass