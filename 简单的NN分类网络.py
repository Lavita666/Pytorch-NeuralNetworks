###########################################
#       创建一个简单的全连接神经网络          #
###########################################

# 导入库
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 超参数设置
num_input = 28*28 #(784)
num_hidden = 200 #隐藏层神经元个数
num_class = 10 #类别数
batch_size = 128 #每次批量数
learning_rate = 0.005 #学习率
num_epochs = 10 #迭代次数

# 载入MNIST数据
Train_dataeset = datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(), download=True)
Test_dataeset = datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(), download=True)
Train_loader = DataLoader(dataset=Train_dataeset,batch_size=batch_size, shuffle=True)
Test_loader = DataLoader(dataset=Test_dataeset,batch_size=len(Test_dataeset), shuffle=False)

# 设置训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 搭建神经网络架构
class NeuralNet(nn.Module):
    def __init__(self,num_input,num_hidden,num_class):
        super(NeuralNet,self).__init__()
        self.fc1=nn.Linear(num_input,num_hidden) # 全连接层； 从输入到隐藏层
        self.fc2=nn.Linear(num_hidden,num_class) # 全连接层； 从隐藏层到输出层
        self.act=nn.ReLU()

    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.fc2(x)
        return x

# 初始化架构
model = NeuralNet(num_input,num_hidden,num_class)

# 定义损失函数和优化器
LossF = nn.CrossEntropyLoss() #此损失函数包含了softmax,因此不需要在输出层另加softmax激活函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
for epoch in range(num_epochs):
    for batch_index, (images, labels) in enumerate(Train_loader):
        images = images.reshape(-1,28*28).to(device) # 64x1x28x28 ---> 64x784
        labels = labels.to(device)
        outputs = model(images)

        #计算损失
        loss = LossF(outputs,labels)

        #梯度的向后传播
        optimizer.zero_grad() #置零
        loss.backward() #向后传
        optimizer.step() #更新

        if batch_index%100==0:
            print('[{}/{}],[{}/{}],loss={:.4f}'.format(epoch,num_epochs,batch_index,len(Train_loader),loss))

# 测试
with torch.no_grad():
    correct_num = 0
    total_num = 0
    for images,labels in Test_loader:
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)#10000x10
        _,predictions = torch.max(outputs,1)
        correct_num = (predictions==labels).sum()
        total_num = (predictions.size(0))
        print("测试集的精度为：{}%".format(correct_num/total_num*100))
