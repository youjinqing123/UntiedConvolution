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

        #self.statistic_f= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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