import torch
import torch.nn as nn
from explore_dataset import train_dataset

#creating a simple cnn model
class simple_cnn(nn.Module):
    def __init__(self):
        super().__init__() 
        self.cov1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1) #in 1,28,28
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cov2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1) #in 32,14,14
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) #out 32,7,7
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=32*7*7,out_features=128)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=128,out_features=10)
    def forward(self,x):
        z1 = self.cov1(x)
        a1 = self.relu1(z1)
        a1_max = self.maxpool1(a1)
        z2 = self.cov2(a1_max)
        a2 = self.relu2(z2)
        a2_max = self.maxpool2(a2)
        a2_max_flatten = self.flatten(a2_max)
        z3 = self.linear1(a2_max_flatten)
        a3 = self.relu3(z3)
        z4 = self.linear2(a3)
        return z4

model = simple_cnn()

#A sample output of the model is a tensor of shape
image,label = train_dataset[0]
y_hat = model(image.unsqueeze(0))
print(y_hat.shape)




    






    




