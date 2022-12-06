import torch
#torch 求导和backward

#求的是一个标量
x = torch.tensor([1.0,2.0],requires_grad=True)
y = (x + 2)**2
z = torch.mean(y)
z.backward()
print(x.grad)

#代表式子 （（x1+2）^2 + (x2 + 2)^20 / 2 

#求的是一个张量
x = torch.tensor([1.0,2.0,3.0],requires_grad=True)
y = (x + 2)**2
z = 4*y
z.backward(torch.tensor([10,1,1]))
print(x.grad)
#tensor([1080.,  192.,  300.])
#代表式子z = 4*(（x1 +2)^2 +（x2 +2)^2 +（x3 +2)^2 )
#z 里面的张亮是x的系数
# 等于的是z* x的导数系数 代入x


