import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import csv

from model import Resnet,CNN
from dataset import TestDataset
from metric import supervisedContrasLoss,KNN
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



def finetune(model,num_epochs,savepath):
    model.train()
    total_qry_acc=0
    for task,(sup_images,sup_labels,qry_images,qry_labels)in enumerate(testloader):
        model.load_state_dict(torch.load(savepath))
        acc = []
        sup_images = sup_images[0].to(device)  # 25x3x84x84
        sup_labels = sup_labels[0].to(device)  # 25
        qry_labels=qry_labels[0]
        for epoch in (bar:=tqdm(range(num_epochs),ncols=0)):
            sup_images1 = transform(sup_images)
            sup_images2 = transform(sup_images)
            sup_images_cat = torch.cat((sup_images1, sup_images2), 0)
            sup_labels_cat = torch.cat((sup_labels, sup_labels), 0)

            optimizer.zero_grad()

            output = model(sup_images_cat)

            loss = supervisedContrasLoss(output, sup_labels_cat)

            loss.backward()
            optimizer.step()

            accuracy = KNN(output, sup_labels_cat,  Ks=[8, 10])
            acc.append(accuracy)
            
            bar.set_description(f'tasks[{task+1}/600]|epoch[{epoch+1:3d}/{num_epochs}]|Finetuning')
            bar.set_postfix_str(
                f' loss {loss.item():.4f} knn acc {sum(acc)/len(acc) :.4f} ')
            scher.step()
            
            
        pred=distance(model,qry_images,sup_images ,sup_labels,qry_labels)
        
        qry_acc=np.sum(pred==qry_labels.numpy())/25
        print(qry_acc)
        # if qry_acc>0.8:
        #     exit()
        total_qry_acc+=qry_acc
    return total_qry_acc/600
            
        



@torch.no_grad()
def distance(model,qry_image,sup_images, sup_labels,qry_labels):
    model.eval()
    # cla_list = []

    sup_images = sup_images.to(device)  # 25x3x84x84
    sup_labels = sup_labels.to(device)  # 25
    qry_image = qry_image[0].to(device)  # 25x3x84x84
    qry_labels=qry_labels.cpu().numpy()

    sup_output = model(sup_images)  # 25xdim
    cla_mean = np.mean([sup_output.cpu().numpy()[
                    (sup_labels == c).cpu().numpy()] for c in range(5)], axis=1)  # 5xdim

    qry_output = model(qry_image).cpu().numpy()  # 25xdim

    # data_np_sup=sup_output.cpu().numpy()
    # # data_np_sup_mean=cla_mean
    # data_np_qry=qry_output
    # tsne = TSNE(n_components=2, random_state=42,perplexity=20)

    # data_=np.vstack([data_np_sup,data_np_qry])
    # # print(data_.shape)
    # data_tsne=tsne.fit_transform(data_)
    # data_tsne_sup=data_tsne[:25]
    # # print(data_tsne_sup.shape)
    # # data_tsne_sup_mean = tsne.fit_transform(data_np_sup_mean)
    # data_tsne_qry = data_tsne[25:]
    # # print(data_tsne_qry.shape)
    # color_dict = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple'}

    # colors = [color_dict[label] for label in qry_labels]
    # # colors_sup_mean = [color_dict[label] for label in np.arange(0,5)]
    # plt.scatter(data_tsne_sup[:, 0], data_tsne_sup[:, 1], c=colors,linewidths=2,edgecolors='k')#linewidths=2,edgecolors='k'
    # plt.scatter(data_tsne_qry[:, 0], data_tsne_qry[:, 1], c=colors)
    # # plt.scatter(data_tsne_sup_mean[:, 0], data_tsne_sup_mean[:, 1], c=colors_sup_mean,edgecolors='k')
    # plt.title('t-SNE Visualization')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.show()
    # plt.savefig('tsne_visualization.png')
    # plt.clf()
    dis = np.sum((qry_output[None, :, :]-cla_mean[:, None, :])**2, axis=-1)

    cla = np.argmin(dis, axis=0)  # 25
    #sim=qry_output@cla_mean.T#25X5
    
    #cla=np.argmax(sim,axis=-1)
    # cla_list.extend(cla)
    # print(cla.shape)
    return cla




if __name__=='__main__':
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(30, [0.2, 0.2], [
                            0.8, 1.2], ),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    # transforms.ColorJitter(0.3,0.3,0.3,0.3),
    ])
    testdata = TestDataset()
    testloader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=True)

    model =CNN()
    # model.load_state_dict(torch.load('student_best.pt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=1e-4)

    scher=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=500,eta_min=5e-5)
    savepath = "student_best_6811.pt"
    

    num_epochs=500
    
    acc=finetune(model,num_epochs,savepath)

    print('acc:',acc)