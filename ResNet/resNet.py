import torch
import torch.nn as nn


"""
Down sampling is done directly using a convolutional layer with a stride of 2 instead of using pooling

In Resnets the number of channels outside a block is always 4 times of it's value when it entered

Residual BLOCKS
=================
* padding -> 0
* stride -> 1
* dowsample using CONV2d with stride 2
* After each Convolution there is a BatchNorm Layer
* Optimization with SGD and MINIBATCH 256
* Learning rate initially 0.1 and decays by 1/10 
* weight decay 0.0001 and momentum of 0.9
augmentations used : RANDOM FLIP and PIXEL NORMALIZATION

"""

class ResidualBlock(nn.Module):
    # Intermediate channels are the channels between the layers
    # Identity Downsampling is needed so that the shape of the activatin is altered so it can be added later on
    def __init__(self,input_channel,intermediate_channels,identity_downsample=None,stride=1):
        super(ResidualBlock,self).__init__()
        self.expansion = 4

        self.conv1 = nn.Conv2d(input_channel,intermediate_channels,kernel_size=1,stride=1,padding=0)
        self.batchNorm_1 = nn.BatchNorm2d(intermediate_channels)

        self.conv2 = nn.Conv2d(intermediate_channels,intermediate_channels,kernel_size=3,stride=stride,padding=1)
        self.batchNorm_2 = nn.BatchNorm2d(intermediate_channels)

        self.conv3 = nn.Conv2d(intermediate_channels,intermediate_channels*self.expansion,kernel_size=1,stride=1,padding=0)
        self.batchNorm_3 = nn.BatchNorm2d(intermediate_channels*self.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride


    def forward(self,x):
        # In this scenario x is the activation from the previouse layer
        identity = x.clone()
        x = self.conv1(x)
        x = self.batchNorm_1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm_2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batchNorm_3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

# + Putting it all together
class ResNet(nn.Module):

    def __init__(self,residual_block,layers,image_channels,num_classes):
        super(ResNet,self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=image_channels,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layers(residual_block,layers[0],intermediate_channels=64,stride=1)
        self.layer2 = self._make_layers(residual_block,layers[1],intermediate_channels=128,stride=2)
        self.layer3 = self._make_layers(residual_block,layers[2],intermediate_channels=256,stride=2)
        self.layer4 = self._make_layers(residual_block,layers[3],intermediate_channels=512,stride=2)

        self.average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512*4,num_classes)


    def forward(self,x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x  = self.layer1(x)
        x  = self.layer2(x)
        x  = self.layer3(x)
        x  = self.layer4(x)

        x  = self.average_pool(x)
        x  = x.reshape(x.shape[0],-1)
        x  = self.fc1(x)
        return x


#   Function to make block layers

    def _make_layers(self,residual_block,num_residual_blocks,intermediate_channels,stride):

        identity_downsample = None
        layers = []

        if stride!=1 or self.in_channels != intermediate_channels*4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels*4,
                    kernel_size=1,
                    stride=stride),
                nn.BatchNorm2d(intermediate_channels*4)
            )

        layers.append(ResidualBlock(self.in_channels,intermediate_channels,identity_downsample,stride))

        self.in_channels = intermediate_channels*4

        #-1 because we have already appended the first block
        for i in range(num_residual_blocks -1):
            layers.append(
                residual_block(self.in_channels,intermediate_channels)
            )

        return nn.Sequential(*layers)


def Resnet50(img_channel=3,num_classes = 1000):
    return ResNet(ResidualBlock,[3,4,6,3],img_channel,num_classes)


if __name__ == "__main__":
    net = Resnet50(img_channel=3,num_classes=1000)
    model_rand_input = torch.randn(4,3,224,224)
    output = net(torch.randn(4,3,224,224)).to("cuda")
    print(output.size())





