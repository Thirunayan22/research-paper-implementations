import torch
import  torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100 , ImageNet
from torch.utils.data import  DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.cuda as cuda
from tqdm import tqdm
import time

device = "cuda"
"""
FIXED HYPERPARAMS
===============================
pooling size:2 
pooling stride : 2
kernel_size : (3,3)
input_size : (224,224,3)
"""

VGG16 = [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"]
#FC LAYERS : 4096->4096->1000 (depends on number of classes) - > softmax

class VGGModel(nn.Module):
    def __init__(self,VGG_type,input_chan=3,num_classes=1000):
        super(VGGModel,self).__init__()
        self.vgg_type = VGG_type
        self.input_chan  = input_chan
        self.num_classes = num_classes

        self.conv_architecture = self.create_architecture(VGG16)
        self.fcl = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
        )

    def create_architecture(self,model_arch) -> nn.Sequential:

        model_seq = []
        input_channels = self.input_chan
        for x in model_arch:
            if type(x) is int:
                output_channels = x
                model_seq += [nn.Conv2d(in_channels= input_channels,out_channels=x, kernel_size=3,stride=1,padding=1),
                              nn.BatchNorm2d(x),
                              nn.ReLU(inplace=True)]

                input_channels = x
            elif type(x) is str:
                model_seq += [nn.MaxPool2d(kernel_size=2,stride=2)]

        return nn.Sequential(*model_seq)

    def forward(self,x):
        x = self.conv_architecture(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fcl(x)
        return x


cuda.empty_cache()
vgg_model = VGGModel(VGG16,num_classes=100).to(device)
input_sample = torch.randn(1,3,32,32).to(device)

x = vgg_model(input_sample)
#Loading Dataset
transform = transforms.Compose([
    transforms.ToTensor(),

])

cifar_100 = CIFAR100("dataset/",train=False,download=True,transform=transform)
dataloader = DataLoader(cifar_100,batch_size=8,shuffle=True)

#hyperparams and optimization parameters
EPOCHS = 100
batch_size = 8
learning_rate  = 0.001

criterion = nn.CrossEntropyLoss()
optim = optim.Adam(vgg_model.parameters(),lr=learning_rate)

for epoch in tqdm(range(EPOCHS)):
    for images,labels in dataloader:

        optim.zero_grad()
        outputs = vgg_model(images.to(device))
        loss = criterion(outputs.detach().cpu(),labels)
        loss.backward()
        optim.step()

        running_loss = loss.item()


print("Completed Model Training!\n")
torch.save(vgg_model,"./vgg16_model.py")