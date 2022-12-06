import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000) #用的是上一个写的生成数据

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
#数据迭代器

from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
#网络层 只有一个线性

#net[0].weight.data.normal_(0, 0.01)
#net[0].bias.data.fill_(0)
#初始化参数 不初始化也无所谓

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 30
l = loss(net(features), labels) #看loss
print(f'epoch {0}, loss {l:f}')
for epoch in range(num_epochs): #训练3个周期
    for X, y in data_iter: #拿数据
        l = loss(net(X) ,y) #算loss
        trainer.zero_grad() #梯度清零
        l.backward()
        trainer.step() #更新参数
    l = loss(net(features), labels) #看loss
    print(f'epoch {epoch + 1}, loss {l:f}')