import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

model_save_path = './models/fc_trained_model.pth'

class FC_Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 100)
        self.fc2 = torch.nn.Linear(100, 50)
        self.fc3 = torch.nn.Linear(50, 32)
        self.fc4 = torch.nn.Linear(32, 10)
        self.fc5 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 8, 3)
        
        self.fc1 = torch.nn.Linear(8 * 5 * 5, 32)
        self.fc2 = torch.nn.Linear(32, 10)
        self.fc3 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test_model(index):
    image, label = training_data.__getitem__(index)
    image = torch.unsqueeze(image, 0)
    # print(image.shape)
    print(f'True label: {label}')
    plt.imshow(image[0][0], cmap = 'gray')
    
    pred = net(image) 
    print(f'Pred label: {round(pred.item())}')
    plt.show()

if __name__ == '__main__':
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    batch_size = 8
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    net = FC_Net()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), .001)

    tb = SummaryWriter()

    epochs = 40
    num_samples = len(train_dataloader.dataset)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        total_loss = 0
        for batch, (X,y) in enumerate(train_dataloader):
            pred = net(X)
            y = torch.reshape(y, (-1,1))
            y = torch.tensor(y, dtype=torch.float32)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * len(X)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (batch + 1) % 1000 == 0:
                print(f'Average loss for batch #{batch+1}: {loss}')

        val = total_loss/num_samples
        tb.add_scalar('Average Loss per Sample', val, epoch + 1)
        print(f'Epoch #{epoch +1}, average sample loss = {val}')

    torch.save(net, model_save_path)

