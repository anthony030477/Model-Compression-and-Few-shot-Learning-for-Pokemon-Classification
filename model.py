import torch
from torchvision.models import resnet18,ResNet18_Weights,regnet_y_128gf,RegNet_Y_128GF_Weights
import torch.nn as nn
from torchsummary import summary
class Regnet(torch.nn.Module):
    def __init__(self):
        super(Regnet, self).__init__()
        self.regnet = regnet_y_128gf(weights=RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_LINEAR_V1)
    def forward(self, x):
        x=self.regnet(x)
        feature = x
        return feature
class Resnet(torch.nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        #self.resnet50.fc = torch.nn.Linear(512, 32)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)

        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)

        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        # feature=x
        # x = self.resnet.fc(x)
        # feature=x
        #x = torch.sigmoid(x)
        # x = nn.Softmax(x / 5)
        feature = x
        return feature

class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.cnn1=nn.Sequential(
                nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(28),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                
            )
            self.cnn2=nn.Sequential(
                nn.Conv2d(28, 56, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(56),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                
            )
            self.cnn3=nn.Sequential(
                nn.Conv2d(56, 136, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(136),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                
            )
            

            
            self.fc = nn.Linear(1224,512)
            
            
        def forward(self, x):
            x=self.cnn1(x)
            
            x=self.cnn2(x)

            x=self.cnn3(x)
            

            x = x.reshape(x.size(0), -1)
            
            x=self.fc(x)
            

            return x
if __name__=='__main__':
    model=CNN()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    summary(model, input_size=(3,28 , 28))
    # x=torch.randn((1,3,64,64))
    # y=model(x)
    # print(y.shape)