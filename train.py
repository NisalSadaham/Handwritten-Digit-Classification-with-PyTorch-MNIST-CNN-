import torch
from torch import nn
import torch.optim
import copy
from dataset import train_loader, validation_loader
from model import simple_cnn

train_losses = []
train_accs = []
val_losses = []
val_accs = []


#creating a instance of the CNN model
torch.manual_seed(42) #to make sure the model is the same every time
model = simple_cnn() #created a instance of the CNN model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #created a instance of the optimizer
criterion = nn.CrossEntropyLoss() #created a binary cross entropy loss function
epochs = 5 #no of iterations the model is going to train


#traning loop of the model
for epoch in range(epochs):
    total = 0
    correct = 0
    running_loss = 0 #to record the running loss
    for images,labels in train_loader: #looping over the training batches(64) each
        
        
        model.train() #switch model to traning mode
        model.zero_grad() #rest the gradeint computation

        #--forward prop---
        y_hat = model(images) #predictions
        loss = criterion(y_hat, labels) #calculate loss
        #--back ward prop
        loss.backward() #send the loss backwards 
        optimizer.step() #update the graidents
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += (y_hat.argmax(dim=1) == labels).sum().item()

    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)    
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images_val, lable_val in validation_loader:
            y_hat_val = model(images_val)
            val_loss = criterion(y_hat_val, lable_val)

            val_running_loss += val_loss.item() * images_val.size(0)
            val_total += lable_val.size(0)
            val_correct += (y_hat_val.argmax(dim=1) == lable_val).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
    print(
        f"Epoch [{epoch + 1}/5] | "
        f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
    )
torch.save(model.state_dict(), "mnist_cnn.pth")
print("Training complete! Model saved to mnist_cnn.pth")











