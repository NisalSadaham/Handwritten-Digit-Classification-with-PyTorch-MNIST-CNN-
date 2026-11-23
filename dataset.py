import torch
from torchvision import transforms,datasets #used to convert images to tensors
from torch.utils.data import random_split #used to split the dataset into training and validation sets
from torch.utils.data import DataLoader #used to load the data in batches

#normalizing the images with z score(features) and mean and standard deviation of the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True,download=False,transform=transform
)

test_dataset = datasets.MNIST(
    root='./data', train=False,download=False,transform=transform

)
#splitting the dataset into training and validation sets
generator = torch.Generator().manual_seed(42)#generator to generate the same dataset 
#splitting data into sebset train_subset - 50000 and validation_subset - 10000
train_subset, validation_subset = random_split(train_dataset,[50000,10000],generator=generator)

#creating the dataloaders for training and validation and test data
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True) #shuffles the data to prevent the model from learning based on a fixed ordering of training examples
validation_loader = DataLoader(validation_subset, batch_size=64,shuffle=False) #we put shuffle for val and test for false no need of shuffling
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#grabbing a single batch to inspect the data
#images,labels = next(iter(train_loader))
#print("Image shape:",images.shape)
#print("Label shape:",labels.shape)
#print("Image data type:",images.dtype)
#print("Images max:",images.max())
#print("Image min:",images.min())



