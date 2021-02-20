import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import random
import tqdm
from tqdm import tqdm

def show_sample(dataloader):
    sample = next(iter(dataloader))[0][random.randint(0,32)]
    img = sample.permute(1,2,0)
    plt.imshow(img.squeeze())
    plt.show()



class Lenet(nn.Module):
    def __init__(self):
        super(Lenet,self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.avg_pooling = nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5)
        self.fc1  =  nn.Linear(in_features=120,out_features=84)
        self.output_layer = nn.Linear(in_features=84,out_features=10)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.avg_pooling(x)
        x = self.relu(self.conv2(x))
        x = self.avg_pooling(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0],-1)  #flattening operation
        x = self.relu(self.fc1(x))
        x = self.softmax(self.output_layer(x))
        return x

lenet_model = Lenet()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Pad(2)
]
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lenet_model.parameters(),lr=0.001)

mnist = MNIST(root='datasets',download=True,train=True,transform=transform)
dataloader = DataLoader(mnist,batch_size=16,shuffle=True)
EPOCHS = 50
for epoch in tqdm(range(EPOCHS)):
    for i,data in enumerate(dataloader,0):
        inputs,labels = data
        optimizer.zero_grad()
        outputs = lenet_model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()



print("finished training")
torch.save(lenet_model,"lenet.pth")

# show_sample(dataloader)
# sample_input = torch.randn(32,1,32,32)
