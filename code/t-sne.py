from torchvision import datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from model import Resnet,CNN
from metric import KNN,supervisedContrasLoss
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



transform = transforms.Compose([

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomAffine(30, [0.2, 0.2], [
                            0.8, 1.2], shear=(0, 0, 0, 45)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.ColorJitter(0.3,0.3,0.3,0.3),
])
transform_totensor=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28,28),antialias=True)
])
traindata=datasets.cifar.CIFAR10(root='cifar10',train=True,transform=transform_totensor,download=True)

trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=512, shuffle=True)

model = CNN()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load('student_best_6811.pt'))
def train( model):
    model.eval()
    for images, labels in (bar := tqdm(trainloader,ncols=0)):

        images = torch.cat((images, images), 0)
        images = images.to(device)
        images = transform(images)
        labels=labels.numpy()
        labels=np.hstack([labels,labels])
        output = model(images)

        data_np=output.cpu().detach().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        data_tsne = tsne.fit_transform(data_np)

        color_dict = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple',
              5: 'brown', 6: 'pink', 7: 'gray', 8: 'olive', 9: 'cyan'}
        colors = [color_dict[label] for label in labels]
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=colors)
        plt.title('t-SNE Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()
        plt.savefig('tsne_visualization.png')
        break

train(model)

