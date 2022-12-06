import random
import torch
import matplotlib.pyplot as plt

#生成测试数据
def synthetic_data(w, b, exp_number):
    w= w.reshape((2,1)) #也可以这样穿个2d直接resize好
    X = torch.normal(0, 1, (exp_number, len(w))) # number*len(w) 的矩阵
    y = torch.matmul(X, w) + b #此处是2d 和 1d乘  1d扩展成2d（在后面插一个维度） 满足第一个矩阵列=第二个矩阵行相等相乘 然后将刚刚插得那个维度置为0
    y += torch.normal(0, 0.01, y.shape) #yshape是一维的
    return X, y.reshape(-1, 1) #变成二维的

true_w = torch.tensor([[2, -3.4]])  #size 1, 2
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

#plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
#plt.show()

#读取数据
def data_iter(batch_size, features, labels):
    num_exp = len(labels)
    indices = list(range(num_exp)) #索引
    random.shuffle(indices) #打乱索引
    for i in range(0, num_exp, batch_size):
        batch_indices = indices[i:min(i+batch_size, num_exp)] #这里不用转成tensor做索引也可以
        yield features[batch_indices], labels[batch_indices]

batch_size = 10


#初始化模型参数 不初始化也可以 随便你
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    #print(y_hat.shape, y.shape)
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 #保证形状是一样的 然而本来就是一样的

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():#当你确认不需要反向传播时，这样可以节省空间
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()#每次求导前，要清零，不然会在前面的求导结果上迭代，这里就是

lr = 0.03
num_epochs = 1 
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward() #求和 然后在上面除以了batchsize
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    #这里是看每轮的loss的 也不需要存梯度
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')



