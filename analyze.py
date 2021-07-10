import torch
from model import FC_Net, Net, model_save_path
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from draw import resize_image

model = torch.load('./models/fc_trained_model.pth')

test_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

batch_size = 8
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

def test_model(index):
    image, label = test_data.__getitem__(index)
    image = torch.unsqueeze(image, 0)
    # print(image.shape)
    print(f'True label: {label}')
    plt.imshow(image[0][0], cmap = 'gray')
    
    pred = model(image) 
    print(f'Pred label: {round(pred.item())}')
    plt.show()

def plot_missclassifcations():
    num_correct = 0
    size = len(test_dataloader.dataset)

    misses = []

    for batch, (X,y) in enumerate(test_dataloader):
        preds = model(X)

        for i in range(len(y)):
            if round(preds[i].item()) == y[i].item():
                num_correct += 1
            else:
                # print(f'Correct label was {y[i]}')
                # print(f'Model predicted {round(preds[i].item())}')
                misses.append(y[i].item())

    print('\nNum correct', num_correct)
    print('Num samples', size)
    print('Accuracy:', num_correct / size)

    counts = {}

    for i in range(10):
        counts[i] = 0
        for j in misses:
            if j == i:
                counts[i] += 1

    x_vals = counts.keys()
    y_vals = counts.values()

    plt.bar(x_vals, y_vals)
    plt.title('Misclassifcations per class')
    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('# of misclassifcations', fontsize=14)
    plt.show()

if __name__ == '__main__':
    # image = resize_image('images/user_drawn_image.png')

    # plt.imshow(image[0][0], cmap = 'gray')
    # plt.show()
    for i in range(0, 10000, 1000):
        test_model(i)



