## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-3)/1 +1 = (32,222,222)
        ## after maxpool = (32,111,111)
        self.conv1 = nn.Conv2d(1, 32, 3)
        
        ## output size = (W-F)/S +1 = (111-3)/1 +1 = (48,109,109)
        ## after maxpool = (48,54,54)
        self.conv2 = nn.Conv2d(32, 48, 3)
        
        ## output size = (W-F)/S +1 = (54-5)/1 +1 = (32,50,50)
        ## after maxpool = (32,25,25)
        self.conv3 = nn.Conv2d(48, 32, 5)
        
         
        self.dense1 = nn.Linear(20000,544)
        self.dense2 = nn.Linear(544,136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.maxpool = nn.MaxPool2d(2,2)
        self.dropout_lin = nn.Dropout(p=0.3)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.maxpool(F.elu(self.conv1(x)))
        x = self.maxpool(F.elu(self.conv2(x)))
        x = self.maxpool(F.elu(self.conv3(x)))
        # flatten x
        x = x.view(x.size(0), -1)
        # drop some nodes to increase robustness
        x = self.dropout_lin(x)
        # dense layer
        x = self.dense1(x)
        x = self.dropout_lin(x)
        x = self.dense2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
