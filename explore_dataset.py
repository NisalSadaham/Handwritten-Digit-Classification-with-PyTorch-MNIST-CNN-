import torch
from torchvision import transforms, datasets
from matplotlib import pyplot as plt

#converting the images to tensors
transform = transforms.ToTensor()

#loading traning data
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform,download=True)
#loading test data
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform,download=True)

#display dataset information
print("Training dataset size:", len(train_dataset))
print("Testing dataset size:", len(test_dataset))

# Inspect one sample
image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Label:", label)
print("Image data type:", image.dtype)
print("Image max value:", image.max())
print("Image min value:", image.min())

#plotting the image
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label: {label}')
plt.axis('off')
plt.show()
