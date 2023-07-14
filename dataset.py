from torch.utils.data import Dataset
import torch
import os
from torchvision.io import read_image,ImageReadMode
import torchvision.transforms as transforms
import random
transform = transforms.Compose([
        transforms.Resize((28,28),antialias=True),
    ])

class TestDataset(Dataset):
    def __init__(self, filename='fewshot_data'):
        self.tasks_path=os.listdir(filename)

    def __len__(self):
        return 600

    def __getitem__(self, index):
        task_path=os.path.join('fewshot_data',self.tasks_path[index])
        sup_images=torch.zeros((25,3,28,28))
        sup_labels=torch.zeros((25,))
        qry_images=torch.zeros((25,3,28,28))
        qry_labels=torch.zeros((25,))
        for i in range(5):
            images_path=os.listdir(task_path+'/'+str(i))
            random.shuffle(images_path)
            sup_images_path = images_path[:5]
            qry_images_path = images_path[5:]
            for j in range(5):
                sup_image = read_image(task_path+'/'+str(i)+'/'+sup_images_path[j],ImageReadMode.RGB)/255
                sup_image=transform(sup_image)
                sup_images[i*5+j]=sup_image
                sup_labels[i*5+j]=i
                qry_image = read_image(task_path+'/'+str(i)+'/'+qry_images_path[j],ImageReadMode.RGB)/255
                qry_image=transform(qry_image)
                qry_images[i*5+j]=qry_image
                qry_labels[i*5+j]=i
        return sup_images,sup_labels,qry_images,qry_labels




if __name__=='__main__':
    testdataset=TestDataset()
    for i in range(600):
        print(testdataset[i][1])
    # for i in range(600):
    #     for j in range(5):
    #         images_path=os.listdir('fewshot_data'+'/'+str(i)+'/'+str(j))
    #         for k in images_path:
    #             if k=='naver_done':
    #                 os.remove('fewshot_data'+'/'+str(i)+'/'+str(j)+'/'+k)
    #                 print('remove')