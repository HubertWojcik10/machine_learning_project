import numpy as np

#pytorch imports
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torch.nn.functional as F

#torchmetrics imports 
from torchmetrics.functional import accuracy
from torchmetrics import F1Score

class CustomDataset(Dataset):
    '''
        Custom dataset class used for loading the data from the .npy file
        and preparing it before transforming it into a pytorch dataset
    '''
    def __init__(self, npy_file, transform=None):
        self.data = np.load(npy_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #get the y label (target)
        target = self.data[idx, -1]

        #reshape the image to 28x28
        image = np.reshape(self.data[idx, :-1], (-1, 28))
       
        #create a dictionary with the image and the target
        sample = {'image': image, 'target': target} 

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

class Net(nn.Module):
    '''
        A class that defines the CNN model, and is a subclass of nn.Module
    '''
    def __init__(self):
        super().__init__()
        #convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        #pooling layer (max pooling)
        self.pool = nn.MaxPool2d(2, 2)

        #fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1) 

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def to_pytorch_format(batch_size=4, norm_metric=1/255):
    '''
        A function that uses torchvision functions to transform the data into a pytorch dataset
        and then loads it into a dataloader. Also, it returns the classes of the dataset
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(norm_metric, norm_metric),
        transforms.Grayscale(num_output_channels=1)])
        
    #create the train_loader and apply the above transformations
    train_set = CustomDataset(train_path, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    #create the test_loader and apply the above transformations
    test_set = CustomDataset(test_path, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    #define the classes of the dataset
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Shirt']

    return train_loader, test_loader, classes


def train_cnn_model(model, train_loader, test_loader, epochs_num=10, testing=False, lr=0.001, momentum=0.9):
    '''
        A function that trains the CNN model using the training data
    '''
    #model.to(device)

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs_num):
        current_loss = 0.0 #to keep track of the model loss

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

            current_loss += loss.item()
            if testing and i % 2000 == 1999:   
                print(f'epoch: {epoch+1}, loss: {current_loss / 2000:.3f}')
                current_loss = 0.0

    if testing: print('Finished training')

def validate_cnn_model(model):
    '''
        A function that validates the CNN model using the test data
    '''

    predictions , targets= [], [] #used to compute accuracy and f1 score

    # torch.no_grad() makes sure that we don't track the gradients as we are not training
    with torch.no_grad():
        for data in test_loader:
            images, labels = Variable(data['image']), Variable(data['target'])

            # calculate outputs by running images through the network
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            predictions.append(predicted)
            targets.append(labels)


    #turn predictions and targets into tensors
    predictions, targets = torch.cat(predictions), torch.cat(targets)
    acc = accuracy(predictions, targets, task='multiclass', num_classes=5) * 100
    f1 = F1Score(num_classes=5, average='macro', task='multiclass') 
    f1_score = f1(predictions, targets) * 100

    print(f'Test data accuracy: {acc:.2f}%, F1 score: {f1_score:.2f}%')


if __name__ == '__main__':
    #device = torch.device('mps')
    train_path, test_path = 'fashion_train.npy', 'fashion_test.npy'

    #hyperparameters
    EPOCHS_NUM = 10
    BATCH_SIZE = 4
    TESTING = True
    LR = 0.001
    MOMENTUM = 0.7

    train_loader, test_loader, classes = to_pytorch_format(batch_size=BATCH_SIZE)
    model = Net()

    train_cnn_model(model, train_loader, test_loader, testing=TESTING, epochs_num=EPOCHS_NUM,
                    lr=LR, momentum=MOMENTUM)
    validate_cnn_model(model)