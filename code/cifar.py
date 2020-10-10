import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from UConv import UntiedConv


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

'''
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()


# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.uc1 = UntiedConv((3, 32, 32), 3, 6, (5, 5), 28*28, 0, 1)
        self.pc1 = nn.Conv2d(3, 6, 5)
        self.wl1 = nn.Conv2d(3, 1, 5)

        self.uc2 = UntiedConv((6, 14, 14), 6, 16, (5, 5), 10 * 10, 0, 1)
        self.pc2 = nn.Conv2d(6, 16, 5)
        self.wl2 = nn.Conv2d(6, 1, 5)

        self.is_test=False
        self.statistic = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



        '''
         def __init__(self,input_shape ,in_channels ,out_channels, kernel_size ,kernel_num ,padding=0, stride=1):
        super(UntiedConv, self).__init__()
        self.input_shape =  input_shape  # (3,10,12) (16,14,14)
        self.in_channels =  in_channels  # (3) (16)
        self.out_channels =  out_channels  # (2) (32)
        self.kernel_size = kernel_size  # (4,5) (5,5)
        self.kernel_num = kernel_num  # (56) (100)
        self.padding = padding
        self.stride = stride
        '''

    def forward(self, x):

        x=self.conv1(x)
        #x=self.uc(x)
        '''
        x1 = self.uc1(x)
        x2 = self.pc1(x)
        x3 = self.wl1(x)
        x4 = 1 - x3

        x = x1 * x3 + x2 * x4
        '''

        x = self.pool(F.relu(x))

        #x=self.conv2(x)

        x1 = self.uc2(x)
        x2 = self.pc2(x)
        self.x3 = F.sigmoid(self.wl2(x))
        #print(self.x3.mean())
        x4 = 1 - self.x3

        if self.is_test==True:
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

        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

'''
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        print(i)
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


torch.save(net.state_dict(), './u_b1_cifar.pth')
print('Finished Training')
'''

net = Net()
net.load_state_dict(torch.load('./u_b1_cifar.pth'))
net.eval()
'''
dataiter = iter(testloader)
images, labels = dataiter.next()




# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''
'''
i=0
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        if i==200:
            break
        images, labels = data


        #print(labels)
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)


        a=''.join('%5s' % classes[labels[j]] for j in range(1))
        print(a)


        if a == 'truck':
            print("in test")
            print(images.shape)
            print(net.x3.shape)
            print(' '.join('%5s' % classes[labels[j]] for j in range(1)))
            print(' '.join('%5s' % classes[predicted[j]] for j in range(1)))
            imshow(torchvision.utils.make_grid(images))
            imshow(torchvision.utils.make_grid(net.x3))

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        i+=1

print('Accuracy of the network on the 100 test images: %d %%' % (
    100 * correct / total))



'''
i=0
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        if i==100:
            break
        images, labels = data


        #print(labels)
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        if i==29:
            print("in test")
            print(images.shape)
            print(net.x3.shape)
            print(' '.join('%5s' % classes[labels[j]] for j in range(1)))
            print(' '.join('%5s' % classes[predicted[j]] for j in range(1)))
            imshow(torchvision.utils.make_grid(images))
            imshow(torchvision.utils.make_grid(net.x3))

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        i+=1

print('Accuracy of the network on the 100 test images: %d %%' % (
    100 * correct / total))
'''


'''

correct = 0
total = 0
net.is_test=True
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


print(net.statistic)





