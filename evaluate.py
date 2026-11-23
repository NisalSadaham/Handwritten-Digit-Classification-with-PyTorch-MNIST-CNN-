from dataset import test_loader
from model import simple_cnn
import torch
from torch import nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

criterion = nn.CrossEntropyLoss()   

model = simple_cnn()
model.load_state_dict(torch.load("mnist_cnn.pth", weights_only=True))
model.eval()

all_preds = [] 
all_labels = []
total_loss = 0.0

with torch.no_grad():
    for images_val, label_val in test_loader:
        y_hat_test = model(images_val)

        # weight each batch by its size so the final average is exact
        total_loss += criterion(y_hat_test, label_val).item() * len(label_val) #muliplying by the len so it will not be the everage loss here since we need to add the raw losses 

        all_preds.append(y_hat_test.argmax(dim=1))
        all_labels.append(label_val)

preds = torch.cat(all_preds) #cat is concatenate in here it is used to join all the perdiction tensors to a single tensor
labels = torch.cat(all_labels)

test_loss = total_loss / len(test_loader.dataset) #average of loss is calculated here
test_acc = accuracy_score(labels, preds)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.2%}")


"""#Generate and save the Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(labels, preds, cmap="Blues", ax=ax)
plt.title("MNIST Test Set Confusion Matrix")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.close()

#Find indices where predictions did not match true labels
wrong_idx = (preds != labels).nonzero(as_tuple=True)[0]

# Plot the first 9 misclassified digits in a 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    idx = wrong_idx[i]
    img, _ = test_loader.dataset[idx]
    
    # Undo normalization (mean=0.1307, std=0.3081) so pixels display cleanly
    img = img * 0.3081 + 0.1307
    
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"Pred: {preds[idx].item()} | True: {labels[idx].item()}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("misclassified.png", bbox_inches="tight")
plt.close()

print("Saved confusion_matrix.png and misclassified.png")"""