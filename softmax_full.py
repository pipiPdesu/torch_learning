import torch
from IPython import display
from d2l import torch as d2l
import torchvision
from torch.utils import data
from torchvision import transforms
d2l.use_svg_display() #一定要用不知道为什么

def get_dataloader_workers():
    return 4
    #几线程读图

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

num_inputs = 784
num_outputs = 10
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X) #指数
    partition = X_exp.sum(1, keepdim=True) #按行求和
    return X_exp / partition  # 这里应用了广播机制 归一化吧大概

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y]) #yhat里两个矩阵 左行索引 右列索引

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) #预测最大的概率的索引
    cmp = y_hat.type(y.dtype) == y  
    return float(cmp.type(y.dtype).sum()) #返回的是一个batch里面预测正确的数量

class Accumulator:  #@save
    """在n个变量上累加"""
    #其实就是开了参数个数个列表
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]#zip 两个数组一起迭代 第一个存预测正确 第二个存总数


    def reset(self):
        self.data = [0.0] * len(self.data) #重置

    def __getitem__(self, idx):
        return self.data[idx] #返回下标为多少的矩阵

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())#预测正确和总数
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):#如果传进来的是网络的话
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer): 
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward() #这里求的平均  为什么多这么一步呢
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward() #这里球的和
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(test_acc)

lr = 0.1
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():#当你确认不需要反向传播时，这样可以节省空间
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
def updater(batch_size):
    return sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
