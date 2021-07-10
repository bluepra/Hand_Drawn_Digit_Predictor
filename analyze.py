import torch
from model import Net, model_save_path
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

model = torch.load('models/trained_model.pth')

test_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

batch_size = 8
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

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



