import torch, torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import image_input as im



def create_lenet():
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding = 2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride = 2),

        nn.Conv2d(6, 16, 5, padding = 0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride = 2),

        nn.Flatten(),
        nn.Linear(400 , 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )
    return model

#sprawdzenie jak poprawny jest utworzony model
def validate(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        x = model(images)
        value, pred = torch.max(x, 1)
        total += x.size(0)
        correct += torch.sum(pred ==labels)
    return correct * 100. / total

def train(numb_epoch = 3, lr = 1e-3, device  = "cpu"):
    cnn = create_lenet()
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr = 1e-3)
    max_accuracy = 0
    accuracies = []
    for epoch in range(numb_epoch):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
        accuracy = float(validate(cnn,val_loader))
        accuracies.append(accuracy)
        if(accuracy > max_accuracy):
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
        print("Epoch: ", epoch + 1, "Accuracy: ", accuracy, "%")
    return best_model

def guess_number(filename, model):
    img_to_guess =  im.get_img(filename)
    plt.imshow(img_to_guess, cmap = "gray")
    img_to_guess = img_to_guess.unsqueeze(0)   
    img_to_guess = img_to_guess.unsqueeze(0)
    
    plt.show()
    yb =  model(img_to_guess)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

if(__name__ == "__main__"):
    train_dataset = MNIST(root='data/', train=True,transform=transforms.ToTensor())
    val_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)
    torch.save(train(10),"model.pth")
