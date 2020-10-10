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








transform=transforms.Compose([
    transforms.Resize(64), #缩放图片，保持长宽比不变，最短边的长为224像素,
    transforms.CenterCrop(64), #从中间切出 224*224的图片
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

'''
valDataSet0=DataClass('D:/cb/CS18-validation/background')
valDataSet1=DataClass('D:/cb/CS18-validation/comment')
valDataSet2=DataClass('D:/cb/CS18-validation/decoration')
valDataSet3=DataClass('D:/cb/CS18-validation/main_text')

valDataSet0=DataClass('D:/cb/CS18-public-test/background')
valDataSet1=DataClass('D:/cb/CS18-public-test/comment')
valDataSet2=DataClass('D:/cb/CS18-public-test/decoration')
valDataSet3=DataClass('D:/cb/CS18-public-test/main_text')
'''
'''
valDataSet0=DataClass('D:/cb/CB55-public-test/background')
valDataSet1=DataClass('D:/cb/CB55-public-test/comment')
valDataSet2=DataClass('D:/cb/CB55-public-test/decoration')
valDataSet3=DataClass('D:/cb/CB55-public-test/main_text')


valDataSet0=DataClass('D:/cb/CB55-validation/background')
valDataSet1=DataClass('D:/cb/CB55-validation/comment')
valDataSet2=DataClass('D:/cb/CB55-validation/decoration')
valDataSet3=DataClass('D:/cb/CB55-validation/main_text')
'''

'''
valDataSet0=DataClass('D:/cb/CS863-validation/background')
valDataSet1=DataClass('D:/cb/CS863-validation/comment')
valDataSet2=DataClass('D:/cb/CS863-validation/decoration')
valDataSet3=DataClass('D:/cb/CS863-validation/main_text')


valDataSet0=DataClass('D:/cb/CS863-public-test/background')
valDataSet1=DataClass('D:/cb/CS863-public-test/comment')
valDataSet2=DataClass('D:/cb/CS863-public-test/decoration')
valDataSet3=DataClass('D:/cb/CS863-public-test/main_text')
'''
valDataSet0=DataClass('D:/cb/CS18-public-test/background')
valDataSet1=DataClass('D:/cb/CS18-public-test/comment')
valDataSet2=DataClass('D:/cb/CS18-public-test/decoration')
valDataSet3=DataClass('D:/cb/CS18-public-test/main_text')

whole_data_val=valDataSet0
whole_data_val=whole_data_val.__add__(valDataSet1)
whole_data_val=whole_data_val.__add__(valDataSet2)
whole_data_val=whole_data_val.__add__(valDataSet3)

classes = ('background', 'comment', 'decoration', 'main_text')

'''
trainloader = torch.utils.data.DataLoader(whole_data_train, batch_size=4,
                                          shuffle=True, num_workers=0)
'''
testloader = torch.utils.data.DataLoader(whole_data_val, batch_size=1,
                                         shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BlockNet(nn.Module):
    def __init__(self):
        super(BlockNet, self).__init__()
        self.b1 = MyBlock((3, 64, 64), 3, 4, (5, 5), 20 * 20, 0, 3) #此处大小待定 (4,20,20)
        self.b2 = MyBlock((4, 20, 20), 4, 6, (3, 3),  9* 9, 0, 2) #此处大小待定 (6,9,9)
        self.b3 = MyBlock((6, 9, 9), 6, 16, (3, 3),  4* 4, 0, 2)#此处大小待定 (16,4,4)
        self.b4 = MyBlock((16, 4, 4), 16, 32, (3, 3),  1* 1, 0, 2)#此处大小待定 (32,1,1)
        self.fc1 = nn.Linear(32 * 1 * 1, 4)
        '''
        self.statistic=[0,0,0,0,0,0,0,0,0,0]
        self.statistic_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.statistic_3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.statistic_4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        '''

    def forward(self, x):
        x = F.relu(self.b1(x))
        #print(self.b1.x3)
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))
        x = F.relu(self.b4(x))
        x = x.view(-1, 32 * 1 * 1)
        x = self.fc1(x)

        '''
        if self.b1.x3.mean()<0.1:
            self.statistic[0]+=1
        elif self.b1.x3.mean()>=0.1 and self.b1.x3.mean()<0.2:
            self.statistic[1] += 1
        elif self.b1.x3.mean() >= 0.2 and self.b1.x3.mean() < 0.3:
            self.statistic[2] += 1
        elif self.b1.x3.mean() >= 0.3 and self.b1.x3.mean() < 0.4:
            self.statistic[3] += 1
        elif self.b1.x3.mean() >= 0.4 and self.b1.x3.mean() < 0.5:
            self.statistic[4] += 1
        elif self.b1.x3.mean() >= 0.5 and self.b1.x3.mean() < 0.6:
            self.statistic[5] += 1
        elif self.b1.x3.mean() >= 0.6 and self.b1.x3.mean() < 0.7:
            self.statistic[6] += 1
        elif self.b1.x3.mean() >= 0.7 and self.b1.x3.mean() < 0.8:
            self.statistic[7] += 1
        elif self.b1.x3.mean() >= 0.8 and self.b1.x3.mean() < 0.9:
            self.statistic[8] += 1
        elif self.b1.x3.mean() >= 0.9 and self.b1.x3.mean() <=1:
            self.statistic[9] += 1

        if self.b2.x3.mean() < 0.1:
            self.statistic_2[0] += 1
        elif self.b2.x3.mean() >= 0.1 and self.b2.x3.mean() < 0.2:
            self.statistic_2[1] += 1
        elif self.b2.x3.mean() >= 0.2 and self.b2.x3.mean() < 0.3:
            self.statistic_2[2] += 1
        elif self.b2.x3.mean() >= 0.3 and self.b2.x3.mean() < 0.4:
            self.statistic_2[3] += 1
        elif self.b2.x3.mean() >= 0.4 and self.b2.x3.mean() < 0.5:
            self.statistic_2[4] += 1
        elif self.b2.x3.mean() >= 0.5 and self.b2.x3.mean() < 0.6:
            self.statistic_2[5] += 1
        elif self.b2.x3.mean() >= 0.6 and self.b2.x3.mean() < 0.7:
            self.statistic_2[6] += 1
        elif self.b2.x3.mean() >= 0.7 and self.b2.x3.mean() < 0.8:
            self.statistic_2[7] += 1
        elif self.b2.x3.mean() >= 0.8 and self.b2.x3.mean() < 0.9:
            self.statistic_2[8] += 1
        elif self.b2.x3.mean() >= 0.9 and self.b2.x3.mean() <= 1:
            self.statistic_2[9] += 1

        if self.b3.x3.mean() < 0.1:
            self.statistic_3[0] += 1
        elif self.b3.x3.mean() >= 0.1 and self.b3.x3.mean() < 0.2:
            self.statistic_3[1] += 1
        elif self.b3.x3.mean() >= 0.2 and self.b3.x3.mean() < 0.3:
            self.statistic_3[2] += 1
        elif self.b3.x3.mean() >= 0.3 and self.b3.x3.mean() < 0.4:
            self.statistic_3[3] += 1
        elif self.b3.x3.mean() >= 0.4 and self.b3.x3.mean() < 0.5:
            self.statistic_3[4] += 1
        elif self.b3.x3.mean() >= 0.5 and self.b3.x3.mean() < 0.6:
            self.statistic_3[5] += 1
        elif self.b3.x3.mean() >= 0.6 and self.b3.x3.mean() < 0.7:
            self.statistic_3[6] += 1
        elif self.b3.x3.mean() >= 0.7 and self.b3.x3.mean() < 0.8:
            self.statistic_3[7] += 1
        elif self.b3.x3.mean() >= 0.8 and self.b3.x3.mean() < 0.9:
            self.statistic_3[8] += 1
        elif self.b3.x3.mean() >= 0.9 and self.b3.x3.mean() <= 1:
            self.statistic_3[9] += 1

        if self.b4.x3.mean() < 0.1:
            self.statistic_4[0] += 1
        elif self.b4.x3.mean() >= 0.1 and self.b4.x3.mean() < 0.2:
            self.statistic_4[1] += 1
        elif self.b4.x3.mean() >= 0.2 and self.b4.x3.mean() < 0.3:
            self.statistic_4[2] += 1
        elif self.b4.x3.mean() >= 0.3 and self.b4.x3.mean() < 0.4:
            self.statistic_4[3] += 1
        elif self.b4.x3.mean() >= 0.4 and self.b4.x3.mean() < 0.5:
            self.statistic_4[4] += 1
        elif self.b4.x3.mean() >= 0.5 and self.b4.x3.mean() < 0.6:
            self.statistic_4[5] += 1
        elif self.b4.x3.mean() >= 0.6 and self.b4.x3.mean() < 0.7:
            self.statistic_4[6] += 1
        elif self.b4.x3.mean() >= 0.7 and self.b4.x3.mean() < 0.8:
            self.statistic_4[7] += 1
        elif self.b4.x3.mean() >= 0.8 and self.b4.x3.mean() < 0.9:
            self.statistic_4[8] += 1
        elif self.b4.x3.mean() >= 0.9 and self.b4.x3.mean() <= 1:
            self.statistic_4[9] += 1
        '''
        return x



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = BlockNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


print(net)

net.load_state_dict(torch.load('D:/cb/64_7.pth'))
net.eval()

print(net)
net.to(device)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# val and save model
correct = 0
total = 0

statistic_1=[0,0,0,0,0,0,0,0,0,0]
statistic_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
statistic_3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
statistic_4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


        temp=net.b4.x3.mean().cpu().numpy()

        if predicted==0:
            if temp < 0.1:
                statistic_1[0] += 1
            elif temp >= 0.1 and temp < 0.2:
                statistic_1[1] += 1
            elif temp >= 0.2 and temp < 0.3:
                statistic_1[2] += 1
            elif temp >= 0.3 and temp < 0.4:
                statistic_1[3] += 1
            elif temp >= 0.4 and temp < 0.5:
                statistic_1[4] += 1
            elif temp >= 0.5 and temp < 0.6:
                statistic_1[5] += 1
            elif temp >= 0.6 and temp < 0.7:
                statistic_1[6] += 1
            elif temp >= 0.7 and temp < 0.8:
                statistic_1[7] += 1
            elif temp >= 0.8 and temp < 0.9:
                statistic_1[8] += 1
            elif temp >= 0.9 and temp <= 1:
                statistic_1[9] += 1

        if predicted == 1:
            if temp < 0.1:
                statistic_2[0] += 1
            elif temp >= 0.1 and temp < 0.2:
                statistic_2[1] += 1
            elif temp >= 0.2 and temp < 0.3:
                statistic_2[2] += 1
            elif temp >= 0.3 and temp < 0.4:
                statistic_2[3] += 1
            elif temp >= 0.4 and temp < 0.5:
                statistic_2[4] += 1
            elif temp >= 0.5 and temp < 0.6:
                statistic_2[5] += 1
            elif temp >= 0.6 and temp < 0.7:
                statistic_2[6] += 1
            elif temp >= 0.7 and temp < 0.8:
                statistic_2[7] += 1
            elif temp >= 0.8 and temp < 0.9:
                statistic_2[8] += 1
            elif temp >= 0.9 and temp <= 1:
                statistic_2[9] += 1

        if predicted == 2:
            if temp < 0.1:
                statistic_3[0] += 1
            elif temp >= 0.1 and temp < 0.2:
                statistic_3[1] += 1
            elif temp >= 0.2 and temp < 0.3:
                statistic_3[2] += 1
            elif temp >= 0.3 and temp < 0.4:
                statistic_3[3] += 1
            elif temp >= 0.4 and temp < 0.5:
                statistic_3[4] += 1
            elif temp >= 0.5 and temp < 0.6:
                statistic_3[5] += 1
            elif temp >= 0.6 and temp < 0.7:
                statistic_3[6] += 1
            elif temp >= 0.7 and temp < 0.8:
                statistic_3[7] += 1
            elif temp >= 0.8 and temp < 0.9:
                statistic_3[8] += 1
            elif temp >= 0.9 and temp <= 1:
                statistic_3[9] += 1

        if predicted == 3:
            if temp < 0.1:
                statistic_4[0] += 1
            elif temp >= 0.1 and temp < 0.2:
                statistic_4[1] += 1
            elif temp >= 0.2 and temp < 0.3:
                statistic_4[2] += 1
            elif temp >= 0.3 and temp < 0.4:
                statistic_4[3] += 1
            elif temp >= 0.4 and temp < 0.5:
                statistic_4[4] += 1
            elif temp >= 0.5 and temp < 0.6:
                statistic_4[5] += 1
            elif temp >= 0.6 and temp < 0.7:
                statistic_4[6] += 1
            elif temp >= 0.7 and temp < 0.8:
                statistic_4[7] += 1
            elif temp >= 0.8 and temp < 0.9:
                statistic_4[8] += 1
            elif temp >= 0.9 and temp <= 1:
                statistic_4[9] += 1

'''
        if predicted!=labels and labels.item() == 3:
            print(' '.join('%5s' % classes[labels[j]] for j in range(1)))
            print(' '.join('%5s' % classes[predicted[j]] for j in range(1)))
            imshow(torchvision.utils.make_grid(images.cpu()))
            imshow(torchvision.utils.make_grid(net.b1.x3.cpu()))
            imshow(torchvision.utils.make_grid(net.b2.x3.cpu()))
            imshow(torchvision.utils.make_grid(net.b3.x3.cpu()))
            print(net.b3.x3.cpu().mean())
'''
print(correct / total)

print(statistic_1)
print(statistic_2)
print(statistic_3)
print(statistic_4)

'''
print(net.statistic)
print(net.statistic_2)
print(net.statistic_3)
print(net.statistic_4)
'''

