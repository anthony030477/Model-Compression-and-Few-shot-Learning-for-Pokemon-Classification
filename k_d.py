from torchvision import datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
from model import Resnet,CNN
from metric import KNN,supervisedContrasLoss


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

teacher = Resnet()
teacher.load_state_dict(torch.load('teacher_best_cifar10.pt'))
student=CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher.to(device)
student.to(device)

optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4,weight_decay=1e-4)
scher=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=1000,eta_min=1e-6)
loss_func=torch.nn.KLDivLoss(reduction="batchmean")

T = 5
gamma = 0.5
def train(i, num_epochs, teacher,student):
    teacher.eval()
    student.train()
    acc = []
    loss_ls=[]
    for images, labels in (bar := tqdm(trainloader,ncols=0)):
        images = torch.cat((images, images), 0)
        images = images.to(device)
        images = transform(images)
        labels = labels.to(device)  # batchsize

        labels = torch.cat((labels, labels), 0)
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_output = teacher(images)
        student_output = student(images)
        
        loss = (1 - gamma)*supervisedContrasLoss(student_output, labels) + (gamma)*loss_func(F.log_softmax(student_output / T, dim=1), F.softmax(teacher_output / T))
        loss.backward()
        optimizer.step()
        loss_ls.append(loss.item())
        accuracy = KNN(student_output, labels, Ks=[5,10,50])
        acc.append(accuracy)

        bar.set_description(f'epoch[{i+1:3d}/{num_epochs}]|Training')
        bar.set_postfix_str(
            f' loss {sum(loss_ls)/len(loss_ls):.4f} accuracy {sum(acc)/len(acc) :.4f}')
    scher.step()
    return sum(acc)/len(acc)

num_epochs = 3000
accuu = 0.05
for i in range(num_epochs):
    accu = train(i, num_epochs, teacher,student)
    #print('acc:', accu)
    if accu > accuu:
        torch.save(student.state_dict(), 'student_best.pt')
        accuu = accu
print('best acc :',accuu)