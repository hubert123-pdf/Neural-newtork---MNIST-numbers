from cProfile import label
from numpy import size, test
import torch 
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split 
from torch.utils.data import DataLoader 
import torch.nn as nn 
import torch.nn.functional as F
import image_input as im

    
INPUT_SIZE = 28*28
NUM_CLASSES = 10

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(INPUT_SIZE, NUM_CLASSES)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out    

    def training_step(self, batch):
        images, labels = batch 
        out = self(images) # Generate predictions
        loss = F.cross_entropy(out, labels)
        # Calculate loss
        return loss     

    def validation_step(self, batch):
        images, labels = batch 
        out = self(images) # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        acc = accuracy(out, labels) # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}  

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean() # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean() # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}  

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))     

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(num_epochs,lr , model, train_loader, val_loader, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(num_epochs):
        for batch in train_loader:  #wejścia i dokładne wyjścia 
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    print(yb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

if __name__ == "__main__":

    dataset = MNIST(root = "data/", download = True, transform = transforms.ToTensor())
    test_dataset = MNIST(root = "data/", train = False, transform = transforms.ToTensor())

    train_ds, val_ds = random_split(dataset, [50000, 10000])
    train_loader = DataLoader(train_ds, batch_size = 10, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size = 10, shuffle = True)

    img_to_guess =  im.get_img("number.jpg")
    img_from_database, label = test_dataset[15]
    plt.imshow(img_from_database[0], cmap = "gray")
    plt.show()
    plt.imshow(img_to_guess, cmap = "gray")
    plt.show()

    img_to_guess = img_to_guess.unsqueeze(0)
    
    model = MnistModel()
    fit(2, 0.001, model, train_loader, val_loader)
    print('Predicted:', predict_image(img_to_guess, model))