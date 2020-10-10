from torchvision.models.resnet import resnet18
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision



class UntiedConv(nn.Module):
    def __init__(self,input_shape ,in_channels ,out_channels, kernel_size ,kernel_num ,padding=0, stride=1):
        super(UntiedConv, self).__init__()
        self.input_shape =  input_shape  # (3,10,12) (16,14,14)
        self.in_channels =  in_channels  # (3) (16)
        self.out_channels =  out_channels  # (2) (32)
        self.kernel_size = kernel_size  # (4,5) (5,5)
        self.kernel_num = kernel_num  # (56) (100)
        self.padding = padding
        self.stride = stride



        self.output_height = int((input_shape[1] + 2 * padding - kernel_size[0]) / stride + 1)  # (7) (10)
        self.output_width = int((input_shape[2] + 2 * padding - kernel_size[1]) / stride + 1 ) # (8) (10)

        self.one_matrix = torch.zeros((kernel_num,in_channels * kernel_size[0] * kernel_size[1] * kernel_num))

        len=in_channels * kernel_size[0] * kernel_size[1]
        self.one_line=torch.ones(len)
        for num in range(kernel_num):
            self.one_matrix[num, num * len:num * len + len]=self.one_line


        self.weights = nn.Parameter(
            torch.empty((in_channels * kernel_size[0] * kernel_size[1] * kernel_num, out_channels),
                        requires_grad=True, dtype=torch.float))#(16*5*5*100,32)

        #print(self.output_height)
        #print(self.output_width)
        self.bias = nn.Parameter(torch.empty((1, out_channels, self.output_height, self.output_width),
                                             requires_grad=True, dtype=torch.float))#(1,32,10,10)

        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.xavier_uniform_(self.bias)
        self.num=0
        self.inp_unfold_zero=0
        self.unfold=torch.nn.Unfold(kernel_size, 1, self.padding,self.stride)
        self.fold=torch.nn.Fold((self.output_height, self.output_width), (1, 1))

    def forward(self, input):
        #print(self.num)
        inp = input  # (batch_num,3,10,12) (2,16,14,14)
        #inp_unfold = torch.nn.functional.unfold(inp, self.kernel_size, 1, self.padding,self.stride)  # (batch_num,3*4*5,56) (2,16*5*5,100)
        inp_unfold=self.unfold(inp)
        inp_unfold_trans = inp_unfold.transpose(1, 2)  # (batch,56,3*4*5) (2,100,16*5*5)
        #print("shape now")
        #print(inp_unfold.shape)

        self.inp_unfold_zero=inp_unfold_trans.repeat(1,1,inp_unfold_trans.shape[1])

        #self.inp_unfold_zero=self.inp_unfold_zero.cuda()
        #self.one_matrix=self.one_matrix.cuda()
        #self.inp_unfold_zero = self.inp_unfold_zero* self.one_matrix
        self.inp_unfold_zero=self.inp_unfold_zero.cuda()*self.one_matrix.cuda()



        #result = new_output.type(torch.cuda.FloatTensor)


        out_unfold = self.inp_unfold_zero.matmul(self.weights).transpose(1, 2)  # (batch_num,2,56) (2,32,100)
        #out = torch.nn.functional.fold(out_unfold, (self.output_height, self.output_width), (1, 1))  # (batch,2,7,8) (2,32,10,10)
        out=self.fold(out_unfold)
        out_bias = out + self.bias
        self.num+=1
        return out_bias


class MyBlock(nn.Module):
    def __init__(self,input_shape ,in_channels ,out_channels, kernel_size ,kernel_num ,padding=0, stride=1):
        # 调用Module的初始化
        super(MyBlock, self).__init__()

        # 创建将要调用的子层（Module），注意：此时还并未实现MyBlock网络的结构，只是初始化了其子层（结构+参数）
        self.uc1 = UntiedConv(input_shape, in_channels, out_channels, kernel_size, kernel_num, padding, stride)
        self.pc1 = nn.Conv2d(in_channels, out_channels, kernel_size[0],stride=stride,padding=padding)
        self.wl1 = nn.Conv2d(in_channels, 1, kernel_size[0],stride=stride,padding=padding)

    def forward(self, x):
        # 这里relu与pool层选择用Function来实现，而不使用Module，用Module也可以
        x1 = self.uc1(x)
        x2 = self.pc1(x)
        self.x3 = torch.sigmoid(self.wl1(x))

        #self.x3=0

        #print(x3.shape)
        x4 = 1 - self.x3

        x = x1 * self.x3 + x2 * x4
        return x


model_18 = resnet18(pretrained=True)





transform=transforms.Compose([
    transforms.Resize(224), #缩放图片，保持长宽比不变，最短边的长为224像素,
    transforms.CenterCrop(224), #从中间切出 224*224的图片
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) #标准化至[-1,1]
])

class DataClass(data.Dataset):

    def __init__(self,root):
        #所有图片的绝对路径
        self.root=root
        imgs=os.listdir(root)
        self.transforms = transform
        self.imgs=[os.path.join(root,k) for k in imgs]

    def __getitem__(self, index):
        img_path=self.imgs[index]
        #dog-> 1 cat ->0
        if self.root[-2:]=='nd':
            label=0
        elif self.root[-2:]=='nt':
            label=1
        elif self.root[-2:]=='on':
            label=2
        elif self.root[-2:]=='xt':
            label=3
        else:
            assert False

        pil_img=Image.open(img_path)

        if self.transforms:
            data = self.transforms(pil_img)
            #print(data.shape)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data,label

    def __len__(self):
        return len(self.imgs)



trainDataSet00=DataClass('../data/New_patch64/patch_extractor/CB55-training/background')
trainDataSet01=DataClass('../data/New_patch64/patch_extractor/CB55-training/comment')
trainDataSet02=DataClass('../data/New_patch64/patch_extractor/CB55-training/decoration')
trainDataSet03=DataClass('../data/New_patch64/patch_extractor/CB55-training/main_text')

trainDataSet10=DataClass('../data/New_patch64/patch_extractor/CS18-training/background')
trainDataSet11=DataClass('../data/New_patch64/patch_extractor/CS18-training/comment')
trainDataSet12=DataClass('../data/New_patch64/patch_extractor/CS18-training/decoration')
trainDataSet13=DataClass('../data/New_patch64/patch_extractor/CS18-training/main_text')

trainDataSet20=DataClass('../data/New_patch64/patch_extractor/CS863-training/background')
trainDataSet21=DataClass('../data/New_patch64/patch_extractor/CS863-training/comment')
trainDataSet22=DataClass('../data/New_patch64/patch_extractor/CS863-training/decoration')
trainDataSet23=DataClass('../data/New_patch64/patch_extractor/CS863-training/main_text')



valDataSet00=DataClass('../data/New_patch64/patch_extractor/CB55-validation/background')
valDataSet01=DataClass('../data/New_patch64/patch_extractor/CB55-validation/comment')
valDataSet02=DataClass('../data/New_patch64/patch_extractor/CB55-validation/decoration')
valDataSet03=DataClass('../data/New_patch64/patch_extractor/CB55-validation/main_text')

valDataSet10=DataClass('../data/New_patch64/patch_extractor/CS18-validation/background')
valDataSet11=DataClass('../data/New_patch64/patch_extractor/CS18-validation/comment')
valDataSet12=DataClass('../data/New_patch64/patch_extractor/CS18-validation/decoration')
valDataSet13=DataClass('../data/New_patch64/patch_extractor/CS18-validation/main_text')

valDataSet20=DataClass('../data/New_patch64/patch_extractor/CS863-validation/background')
valDataSet21=DataClass('../data/New_patch64/patch_extractor/CS863-validation/comment')
valDataSet22=DataClass('../data/New_patch64/patch_extractor/CS863-validation/decoration')
valDataSet23=DataClass('../data/New_patch64/patch_extractor/CS863-validation/main_text')

whole_data_train=trainDataSet10
whole_data_train=whole_data_train.__add__(trainDataSet11)
whole_data_train=whole_data_train.__add__(trainDataSet12)
whole_data_train=whole_data_train.__add__(trainDataSet13)


whole_data_val=valDataSet10
whole_data_val=whole_data_val.__add__(valDataSet11)
whole_data_val=whole_data_val.__add__(valDataSet12)
whole_data_val=whole_data_val.__add__(valDataSet13)


classes = ('background', 'comment', 'decoration', 'main_text')

trainloader = torch.utils.data.DataLoader(whole_data_train, batch_size=4,
                                          shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(whole_data_val, batch_size=4,
                                         shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = model_18
#net.conv1=UntiedConv((3, 224,224), 3, 64, (7, 7), 112 * 112, 3, 2)
#net.layer4[1].conv2=UntiedConv((512, 7,7), 512, 512, (3, 3), 7 * 7, 1, 1)
#net.layer1[1].conv2=UntiedConv((64, 56,56), 64, 64, (3, 3), 56 * 56, 1, 1)


net.layer4[1].conv2=MyBlock((512, 7,7), 512, 512, (3, 3), 7 * 7, 1, 1)


s= nn.Sequential(
                  nn.Conv2d(512,64,1),
                  nn.ReLU(),
                  MyBlock((64, 7,7), 64, 64, (3, 3), 7 * 7, 1, 1),
                  nn.ReLU(),
                  nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
                )


net.fc=nn.Linear(512, 4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



net.to(device)

num = 0
time = []
value = []

threshold=0.2

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        #print(i)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)

        #imshow(torchvision.utils.make_grid(inputs.cpu()))


        #print(labels.cpu().numpy())
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        #print("ok")
        loss = criterion(outputs, labels)


        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 50 mini-batches
            print('training loss')
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

            # val and save model
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()


            if (correct / total)>threshold:
                print("save model")
                print('Accuracy of the network on the validation set: %d %%' % (
                    100 * correct / total))
                torch.save(net.state_dict(), './res7.pth')
                threshold=(correct / total)


print('Finished Training')