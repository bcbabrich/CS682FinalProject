# imports
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import random
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F  # useful stateless functions

# We need to wrap `flatten` function in a module in order to stack it
# in nn.Sequential
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


# define our network
# right now, this is just a fully-connected 2 layer network
# neural network code taken from here:
# https://towardsdatascience.com/a-simple-starter-guide-to-build-a-neural-network-3c2cf07b8d7c
class AffineNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
        nn.init.kaiming_normal_(self.fc2.weight)
    
    def forward(self, x):                              # Forward pass: stacking each layer together
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
# define a simple convolutional network
# code taken from second homework set
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        in_channel = 1
        channel_1 = 12
        channel_2 = 8
        
        # first convolutional layer object
        self.conv1 = torch.nn.Conv2d(in_channel, channel_1, (5,5), stride=1, padding=2, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight)
        
        # second convolutional layer object
        self.conv2 = torch.nn.Conv2d(channel_1, channel_2, (3,3), stride=1, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv2.weight)
        
        # fully connected layer
        self.fc = nn.Linear(channel_2*28*28, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)
    
    def forward(self, x):                              # Forward pass: stacking each layer together
        return self.fc(flatten(F.relu(self.conv2(F.relu(self.conv1(x))))))

####### SPLIT TRAINING DATASET FURTHER INTO TRAIN/VAL SPLIT #####
# We need to further split our training dataset into training and validation sets.
# We do this split for every point to randomize the train/val split for better generalizability to the test dataset
# train/val split code taken from here:
# https://am207.github.io/2018spring/wiki/ValidationSplits.html?fbclid=IwAR03zx1ws5ONjk1be7ahhMCMV7R1mbVHfhCMC-RoU-Ys7_s81Q_bCrpob6s
def get_train_val_split(train_dataset, batch_size) :
    # Define the indices
    indices = list(range(len(train_dataset))) # start with all the indices in training set
    split = 10000 # define the split size

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # define our samplers -- we use a SubsetRandomSampler because it will return
    # a random subset of the split defined by the given indices without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    # Create the train_loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                    batch_size=batch_size, sampler=train_sampler)

    # You can use your above batch_size or just set it to 1 here.  Your validation
    # operations shouldn't be computationally intensive or require batching.
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                    batch_size=1, sampler=validation_sampler)
    
    return (train_loader, validation_loader)
    
def runExperiment(points, desired_num_points, auto_manual, network_type, experiment_number) :
    ######### INITIALIAL TRAIN/TEST SPLIT #######
    ## Define our MNIST Datasets (Images and Labels) for training and testing
    train_dataset = dsets.MNIST(root='./data', 
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data', 
                               train=False, 
                               transform=transforms.ToTensor())
    
    # our test set will NOT change for each point, so we can just set the test_loader here
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=1,
                                              shuffle=False)
    
    # set hyperparameters for neural network
    # these are not generated by poisson/random right now, so we set them here, separate from experiment loop below
    input_size = 784                             # The image size = 28 x 28 = 784
    num_classes = 10                             # The number of output classes. In this case, from 0 to 9
    
    ######### RUN ALL POINTS #######
     # shuffle points
    random.shuffle(points)
    
    # variables to hold best point info
    highest_accuracy = 0
    best_learning_rate = 0
    best_batch_size =  0
    phase = None # phase variable for run experiments
    p = 0 # count number of points being used (for code testing purposes)

    # here we run all points on train + val splits of data
    # only run desired_num_points of
    # this takes care of the case where generate_poisson_points
    # returned a list of slightly too many points (< 5)
    for i in range(desired_num_points) :
        # draw next point, convert to hyperparameters
        point = points[i]
        learning_rate = point[0]/1000
        hidden_size = 100
        batch_size = int(round(point[1]))
        num_epochs = 5
        # avoid zero value errors
        if learning_rate == 0 : learning_rate = 0.5
        if batch_size == 0 : batch_size = 1
        
        print('running experiment with point ',point)
        experiment_number += 1
        print('on experiment', experiment_number, '=>',experiment_number/(desired_num_points*3*5),' of the way there')

        # perform a train/val split on training data
        train_loader, validation_loader = get_train_val_split(train_dataset, batch_size)

        # RUN NN WITH GENERATED HYPERPARAMETERS
        # create an instance of our network
        if network_type == 'affine' :
            net = AffineNet(input_size, hidden_size, num_classes)
        elif network_type == 'conv' :
            net = ConvNet(num_classes)
        else :
            print('network type not supported')
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        
        # train!
        for epoch in range(num_epochs):
            for j, (images, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
                if network_type == 'affine' :
                    images = Variable(images.view(-1, 28*28))
                elif network_type == 'conv' :
                    images = np.tile(images, (1,1,1,1))
                    images = Variable(torch.from_numpy(images))   # convert to tensor object
                
                labels = Variable(labels)

                optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
                outputs = net(images)                             # Forward pass: compute the output class given a image
                loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label
                loss.backward()                                   # Backward pass: compute the weight
                optimizer.step()                                  # Optimizer: update the weights of hidden nodes

                
                if j % 200 == 0:                              # Logging
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                         %(epoch, num_epochs, j, len(train_dataset)//batch_size, loss.data))
                
        
        # we're still searching for the best hyperparameter point
        # therefore, we test on the VALIDATION set
        correct = 0
        total = 0
        for images, labels in validation_loader:
            if network_type == 'affine' :
                images = Variable(images.view(-1, 28*28))
            elif network_type == 'conv' :
                images = np.tile(images, (1,1,1,1))
                images = Variable(torch.from_numpy(images))   # convert to tensor object
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
            total += labels.size(0)                    # Increment the total count
            correct += (predicted == labels).sum()     # Increment the correct count
        
        accuracy = int((100 * correct / total))

        # print results
        
        q  = 2 # q corresponds to print frequency
        if p % q == 0 :
            print('point number',p)
            print('accuracy',accuracy)
            print('learning_rate',learning_rate)
            print('batch_size',batch_size)
        

        # grab current best accuracy and corresponding point values
        if accuracy > highest_accuracy :
            highest_accuracy = accuracy
            best_learning_rate = learning_rate
            best_batch_size = batch_size

        p += 1
    
    # print('val testing done. highest accuracy returned was',highest_accuracy)
    
    # I hope this works!
    if auto_manual == 'val' :
        return highest_accuracy, experiment_number
    
    # TEST PHASE OF EXPERIMENT RUN
    # here we run the best point found on our test dataset
    # create one more instance of our network
    if network_type == 'affine' :
        net = AffineNet(input_size, hidden_size, num_classes)
    elif network_type == 'conv' :
        net = ConvNet(num_classes)
    else :
        print('network type not supported')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=best_learning_rate)
    
    experiment_number += 1
    print('on experiment', experiment_number, '=>',experiment_number/(desired_num_points*3*5),' of the way there')
    
    # train one last time
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
            labels = Variable(labels)
            
            #print('got here.')
            
            #print('images before reshape',images.shape)
            
            if network_type == 'affine' :
                images = Variable(images.view(-1, 28*28))
            elif network_type == 'conv' :
                images = np.tile(images, (1,1,1,1))
                images = Variable(torch.from_numpy(images))   # convert to tensor object

            print('images after reshape',images.shape)
            optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
            outputs = net(images)                             # Forward pass: compute the output class given a image
            loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label
            loss.backward()                                   # Backward pass: compute the weight
            optimizer.step()                                  # Optimizer: update the weights of hidden nodes

            
            if (i+1) % 200 == 0:                              # Logging
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                     %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
            
    # test on the TEST set
    # get accuracy for TEST set
    correct = 0
    total = 0
    for images, labels in test_loader:
        if network_type == 'affine' :
            images = Variable(images.view(-1, 28*28))
        elif network_type == 'conv' :
            images = np.tile(images, (1,1,1,1))
            images = Variable(torch.from_numpy(images))   # convert to tensor object
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
        total += labels.size(0)                    # Increment the total count
        correct += (predicted == labels).sum()     # Increment the correct count

    test_accuracy = int((100 * correct / total))
    if auto_manual == 'test' :
        return test_accuracy, experiment_number
    return test_accuracy, (best_learning_rate, best_batch_size), experiment_number
