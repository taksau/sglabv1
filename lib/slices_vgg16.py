import torch
from torch import nn

class Slices_VGG(nn.Module):
    def __init__(self):
        super(Slices_VGG, self).__init__()

        self.l1_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.l1_relu1 = nn.ReLU(inplace=True)
        self.l1_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.l1_relu2 = nn.ReLU(inplace=True)
        self.l1_maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        self.l2_conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.l2_relu1 = nn.ReLU(inplace=True)
        self.l2_conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.l2_relu2 = nn.ReLU(inplace=True)
        self.l2_maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        self.l3_conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.l3_relu1 = nn.ReLU(inplace=True)
        self.l3_conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.l3_relu2 = nn.ReLU(inplace=True)
        self.l3_conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.l3_relu3 = nn.ReLU(inplace=True)
        self.l3_maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        self.l4_conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.l4_relu1 = nn.ReLU(inplace=True)
        self.l4_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.l4_relu2 = nn.ReLU(inplace=True)
        self.l4_conv3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.l4_relu3 = nn.ReLU(inplace=True)
        self.l4_maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        self.l5_conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.l5_relu1 = nn.ReLU(inplace=True)
        self.l5_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.l5_relu2 = nn.ReLU(inplace=True)
        self.l5_conv3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.l5_relu3 = nn.ReLU(inplace=True)

    
    def forward(self, x):
        
        x = self.l1_conv1(x)
        x = self.l1_relu1(x)
        x = self.l1_conv2(x)
        x = self.l1_relu2(x)
        x = self.l1_maxpool(x)

        x = self.l2_conv1(x)
        x = self.l2_relu1(x)
        x = self.l2_conv2(x)
        x = self.l2_relu2(x)
        c2 = self.l2_maxpool(x)

        c3 = self.l3_conv1(c2)
        c3 = self.l3_relu1(c3)
        c3 = self.l3_conv2(c3)
        c3 = self.l3_relu2(c3)
        c3 = self.l3_conv3(c3)
        c3 = self.l3_relu3(c3)
        c3 = self.l3_maxpool(c3)

        c4 = self.l4_conv1(c3)
        c4 = self.l4_relu1(c4)
        c4 = self.l4_conv2(c4)
        c4 = self.l4_relu2(c4)
        c4 = self.l4_conv3(c4)
        c4 = self.l4_relu3(c4)
        c4 = self.l4_maxpool(c4)

        c5 = self.l5_conv1(c4)
        c5 = self.l5_relu1(c5)
        c5 = self.l5_conv2(c5)
        c5 = self.l5_relu2(c5)
        c5 = self.l5_conv3(c5)
        c5 = self.l5_relu3(c5)

        return c5 

    def get_weight(self, model):

        self.l1_conv1.weight.data.copy_(model[0].weight.clone().data)
        self.l1_conv1.bias.data.copy_(model[0].bias.clone().data)
        self.l1_conv2.weight.data.copy_(model[2].weight.clone().data)
        self.l1_conv2.bias.data.copy_(model[2].bias.clone().data)

        self.l2_conv1.weight.data.copy_(model[5].weight.clone().data)
        self.l2_conv1.bias.data.copy_(model[5].bias.clone().data)
        self.l2_conv2.weight.data.copy_(model[7].weight.clone().data)
        self.l2_conv2.bias.data.copy_(model[7].bias.clone().data)

        self.l3_conv1.weight.data.copy_(model[10].weight.clone().data)
        self.l3_conv1.bias.data.copy_(model[10].bias.clone().data)
        self.l3_conv2.weight.data.copy_(model[12].weight.clone().data)
        self.l3_conv2.bias.data.copy_(model[12].bias.clone().data)
        self.l3_conv3.weight.data.copy_(model[14].weight.clone().data)
        self.l3_conv3.bias.data.copy_(model[14].bias.clone().data)

        self.l4_conv1.weight.data.copy_(model[17].weight.clone().data)
        self.l4_conv1.bias.data.copy_(model[17].bias.clone().data)
        self.l4_conv2.weight.data.copy_(model[19].weight.clone().data)
        self.l4_conv2.bias.data.copy_(model[19].bias.clone().data)
        self.l4_conv3.weight.data.copy_(model[21].weight.clone().data)
        self.l4_conv3.bias.data.copy_(model[21].bias.clone().data)

        self.l5_conv1.weight.data.copy_(model[24].weight.clone().data)
        self.l5_conv1.bias.data.copy_(model[24].bias.clone().data)
        self.l5_conv2.weight.data.copy_(model[26].weight.clone().data)
        self.l5_conv2.bias.data.copy_(model[26].bias.clone().data)
        self.l5_conv3.weight.data.copy_(model[28].weight.clone().data)
        self.l5_conv3.bias.data.copy_(model[28].bias.clone().data)
