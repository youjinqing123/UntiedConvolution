"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
torchvision
matplotlib
"""
# library
# standard library
import os
import numpy as np

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
# torch.manual_seed(1)    # reproducible
import matplotlib.pyplot as plt


class testConv(nn.Module):
    def __init__(self):
        super(testConv, self).__init__()

    def forward(self, input):
        print("1")
        out = torch.empty((input.shape[0], input.shape[1] * 2, 10, 10))
        for ba in range(out.shape[0]):
            for c in range(out.shape[1]):
                out[ba, c, :, :] = input[ba, int(c / 2), 0:10, 0:10]
        return out


class UntiedConv(nn.Module):
    def __init__(self, input_shape, in_channels, out_channels, kernel_size, kernel_num, padding=0, stride=1):
        super(UntiedConv, self).__init__()
        self.input_shape = input_shape  # (3,10,12) (16,14,14)
        self.in_channels = in_channels  # (3) (16)
        self.out_channels = out_channels  # (2) (32)
        self.kernel_size = kernel_size  # (4,5) (5,5)
        self.kernel_num = kernel_num  # (56) (100)
        self.padding = padding
        self.stride = stride

        self.output_height = int((input_shape[1] + 2 * padding - kernel_size[0]) / stride + 1)  # (7) (10)
        self.output_width = int((input_shape[2] + 2 * padding - kernel_size[1]) / stride + 1)  # (8) (10)

        self.one_matrix = torch.zeros((kernel_num, in_channels * kernel_size[0] * kernel_size[1] * kernel_num))

        len = in_channels * kernel_size[0] * kernel_size[1]
        self.one_line = torch.ones(len)
        for num in range(kernel_num):
            self.one_matrix[num, num * len:num * len + len] = self.one_line

        self.weights = nn.Parameter(
            torch.empty((in_channels * kernel_size[0] * kernel_size[1] * kernel_num, out_channels),
                        requires_grad=True, dtype=torch.float))  # (16*5*5*100,32)

        # print(self.output_height)
        # print(self.output_width)
        self.bias = nn.Parameter(torch.empty((1, out_channels, self.output_height, self.output_width),
                                             requires_grad=True, dtype=torch.float))  # (1,32,10,10)

        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.xavier_uniform_(self.bias)
        self.num = 0
        self.inp_unfold_zero = 0
        self.unfold = torch.nn.Unfold(kernel_size, 1, self.padding, self.stride)
        self.fold = torch.nn.Fold((self.output_height, self.output_width), (1, 1))

    def forward(self, input):
        # print(self.num)
        inp = input  # (batch_num,3,10,12) (2,16,14,14)
        # inp_unfold = torch.nn.functional.unfold(inp, self.kernel_size, 1, self.padding,self.stride)  # (batch_num,3*4*5,56) (2,16*5*5,100)
        inp_unfold = self.unfold(inp)
        inp_unfold_trans = inp_unfold.transpose(1, 2)  # (batch,56,3*4*5) (2,100,16*5*5)
        # print("shape now")
        # print(inp_unfold.shape)

        self.inp_unfold_zero = inp_unfold_trans.repeat(1, 1, inp_unfold_trans.shape[1])
        self.inp_unfold_zero = self.inp_unfold_zero * self.one_matrix

        '''

        # replace the true by 'if the batch size has changed'
        if self.num==0 or True:
            self.inp_unfold_zero=torch.zeros(inp_unfold.shape[0], inp_unfold.shape[2],
                                      inp_unfold.shape[1] * inp_unfold.shape[2])  # (batch_num,56,3*4*5*56) (2,100,100*400)

        len = inp_unfold.shape[1]
        for ba in range(self.inp_unfold_zero.shape[0]):
            for num in range(self.inp_unfold_zero.shape[1]):
                self.inp_unfold_zero[ba, num, num * len:num * len + len] = inp_unfold_trans[ba, num, :]
        '''

        out_unfold = self.inp_unfold_zero.matmul(self.weights).transpose(1, 2)  # (batch_num,2,56) (2,32,100)
        # out = torch.nn.functional.fold(out_unfold, (self.output_height, self.output_width), (1, 1))  # (batch,2,7,8) (2,32,10,10)
        out = self.fold(out_unfold)
        out_bias = out + self.bias
        self.num += 1
        return out_bias


# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 16
LR = 0.001  # learning rate
DOWNLOAD_MNIST = False

# Mnist digits dataset
if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
         :2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]

'''
training_set = np.load('original_60000.npy')
training_set = training_set.reshape(-1, 28, 28)
training_label = np.load('train_labels.npy')



train_data.train_data = torch.tensor(training_set[0:55000,:,:], dtype=torch.float32)
train_data.train_labels = torch.tensor(training_label[0:55000],dtype=torch.long)

test_x=torch.tensor(training_set[59900:60000,:,:], dtype=torch.float32).view(-1,1,28,28)/255
test_y=torch.tensor(training_label[59900:60000],dtype=torch.long)
'''

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )

        '''
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            #testConv(),
            UntiedConv((16, 14, 14), 16, 32, (5, 5), 100, 0, 1),
            #nn.Conv2d(16, 32, 5, 1, 0),     # output shape (32, 10, 10)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 5, 5)
        )
        '''
        self.uc = UntiedConv((16, 14, 14), 16, 32, (5, 5), 100, 0, 1)
        self.pc = nn.Conv2d(16, 32, 5, 1, 0)
        self.wl = nn.Conv2d(16, 1, 5, 1, 0)

        self.r1 = nn.ReLU()
        self.m1 = nn.MaxPool2d(2)

        self.out = nn.Linear(32 * 5 * 5, 10)  # fully connected layer, output 10 classes

        self.is_test = False
        self.statistic = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)


        x1 = self.uc(x)
        x2 = self.pc(x)
        self.x3 = F.sigmoid(self.wl(x))

        # print(self.x3.shape)

        x4 = 1 - self.x3

        if self.is_test == True:
            if self.x3.mean() < 0.1:
                self.statistic[0] += 1
            elif self.x3.mean() >= 0.1 and self.x3.mean() < 0.2:
                self.statistic[1] += 1
            elif self.x3.mean() >= 0.2 and self.x3.mean() < 0.3:
                self.statistic[2] += 1
            elif self.x3.mean() >= 0.3 and self.x3.mean() < 0.4:
                self.statistic[3] += 1
            elif self.x3.mean() >= 0.4 and self.x3.mean() < 0.5:
                self.statistic[4] += 1
            elif self.x3.mean() >= 0.5 and self.x3.mean() < 0.6:
                self.statistic[5] += 1
            elif self.x3.mean() >= 0.6 and self.x3.mean() < 0.7:
                self.statistic[6] += 1
            elif self.x3.mean() >= 0.7 and self.x3.mean() < 0.8:
                self.statistic[7] += 1
            elif self.x3.mean() >= 0.8 and self.x3.mean() < 0.9:
                self.statistic[8] += 1
            elif self.x3.mean() >= 0.9 and self.x3.mean() <= 1:
                self.statistic[9] += 1

        x = x1 * self.x3 + x2 * x4

        x = self.r1(x)
        x = self.m1(x)

        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x  # return x for visualization


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

'''
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        #print(step)
        #print("input")
        #print(b_x.shape)

        output = cnn(b_x)[0]               # cnn output
        #print(output.shape)
        loss = loss_func(output, b_y)   # cross entropy loss
        #print(loss)

        optimizer.zero_grad()           # clear gradients for this training step

        #print('here1')
        loss.backward()                 # backpropagation, compute gradients
        #print('here2')
        optimizer.step()                # apply gradients

        with torch.no_grad():
            if step % 50 == 0:


                accuracy=0


                for i in range(40):
                    #print("i")
                    #print(i)
                    test_x_ba=test_x[i*50:(i+1)*50,:,:,:]
                    #print(test_x_ba.shape)
                    test_output_ba, last_layer_ba = cnn(test_x_ba)
                    pred_y_ba = torch.max(test_output_ba, 1)[1].data.numpy()
                    accuracy += float((pred_y_ba == test_y[i*50:(i+1)*50].data.numpy()).astype(int).sum())

                accuracy/= float(test_y.size(0))

                if accuracy >= 0.97:
                    torch.save(cnn.state_dict(), './u_b0_mnist.pth')
                    print('can finish')
                    exit()


                #print(cnn.x3.mean())
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
'''

net = CNN()
net.load_state_dict(torch.load('./u_b0_mnist.pth'))
net.eval()

net.is_test = True

'''
with torch.no_grad():
        accuracy = 0

        for i in range(80):
            print(i)
            # print("i")
            # print(i)
            test_x_ba = test_x[i * 25:(i + 1) * 25, :, :, :]
            # print(test_x_ba.shape)
            test_output_ba, last_layer_ba = net(test_x_ba)
            pred_y_ba = torch.max(test_output_ba, 1)[1].data.numpy()
            print(pred_y_ba)
            print(test_y[i * 25:(i + 1) * 25])
            accuracy += float((pred_y_ba == test_y[i * 25:(i + 1) * 25].data.numpy()).astype(int).sum())

        accuracy /= float(test_y.size(0))
        print(accuracy)
        print(net.statistic)
'''

test_x_plt = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[2001:] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y_plt = test_data.test_labels[2001:]

print(test_y_plt.shape)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


'''
with torch.no_grad():
    accuracy = 0

    for i in range(100):
        print(i)
        # print("i")
        # print(i)
        test_x_ba = test_x_plt[i * 1:(i + 1) * 1, :, :, :]
        # print(test_x_ba.shape)
        test_output_ba, last_layer_ba = net(test_x_ba)
        pred_y_ba = torch.max(test_output_ba, 1)[1].data.numpy()
        print(pred_y_ba)
        print(test_y_plt[i * 1:(i + 1) * 1])
        imshow(torchvision.utils.make_grid(test_x_ba))
        imshow(torchvision.utils.make_grid(net.x3))
        accuracy += float((pred_y_ba == test_y_plt[i * 1:(i + 1) * 1].data.numpy()).astype(int).sum())

    accuracy /= float(test_y_plt.size(0))
    print(accuracy)
'''

with torch.no_grad():
    accuracy = 0

    for i in range(7990):
        print(i)
        # print("i")
        # print(i)
        test_x_ba = test_x_plt[i * 1:(i + 1) * 1, :, :, :]
        # print(test_x_ba.shape)
        test_output_ba, last_layer_ba = net(test_x_ba)
        pred_y_ba = torch.max(test_output_ba, 1)[1].data.numpy()
        print(pred_y_ba)
        print(test_y_plt[i * 1:(i + 1) * 1])
        accuracy += float((pred_y_ba == test_y_plt[i * 1:(i + 1) * 1].data.numpy()).astype(int).sum())


    #accuracy /= float(test_y_plt.size(0))
    accuracy /= float(7990)
    print(accuracy)




# print 10 predictions from test data
test_output, _ = net(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(net.statistic)
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')