## 1. 线性回归是什么？
线性回归就是通过学习一个线性关系模型来尽可能准确地预测或估计新的输出值。  
  
线性回归是基于[线性模型](Linear-model.md)，**学得 $f(x_i)=wx_i+b$ ，尽可能使得 $f(x_i)$ 逼近于 $y_i$ 。**    
为了达到这个目标，就要**确定 $w$ 和 $b$  ,能使得 $f(x_i)$ 和 $y_i$ 之间的误差最小**。  
这个误差的度量方式<!-- TODO-[度量方式|距离度量] -->有很多，一般用到**均方误差**，即为：  
$$
(w^*,b^*)=\arg{\min_{(w,b)}}\sum_{i=1}^m(y_i-wx_i-b)^2
$$
而求解这个均方误差又使用到了[[最小二乘法]]。最后求得
$$w=\frac{\sum_{i=1}^{m}y_i(x_i-\bar{x})}{\sum_{i=1}^mx_i^2-\frac{1}{m}(\sum_{i=1}^mx_i)^2}$$
$$b=\frac{1}{m}\sum_{i=1}^m(y_i-wx_i)$$
这个误差的量化就是[损失函数（loss function）](Loss-Function-and-Error.md)。  
在求解的过程中会使用[梯度下降法](Gradient-Descent.md)来不断在损失函数递减的方向上更新参数来降低误差，以求得解。  
- 线性回归模型是一个简单的神经网络
<img src="https://zh-v2.d2l.ai/_images/singleneuron.svg" alt="单层神经网络" width="250" height="150">



## 2. 线性回归的拓展
- **多元线性回归(multivariate linear regression)**：样本由多个属性描述的。即为 $f(x_i)= \pmb{w}^T\pmb{x}_i+b, 使得f(\pmb(x_i)\simeq{y_i}$ 
- [[logistic regression|对数几率回归]]：是一种广义的线性回归，用于处理二分类问题，使用逻辑函数将线性组合映射到概率值。

## 3. 线性回归的实现
线性回归模型的实现可以总结为：
1. 初始化参数
2. 前向传播
3. 计算损失
4. 反向传播
5. 重复迭代
6. 模型预测


- 从零开始实现
```python
import random
import torch

# 生成数据集
def synthetic_data(w, b, num_examples):
	"""生成y=Xw+b+噪声"""
	x = torch.normal(0, 1, (num_examples, lern(w)))
	y = torch.matmul(X, w) + b
	y += torch.normal(0, 0.01, y.shape)
	return X, y.shape((-1, 1))

# 读取数据
def data_iter(batch_size, features, labels):
	num_example = len(features)
	indices = list(range(num_examples))
	# 将样本的索引随机打乱，使得迭代获得的批量数据是随机的
	random.shuffle(indices)
	for i in range(0, num_example, batch_size)
		#  选择当前批量的样本索引
		batch_indices = torch.tensor(indices[i:min(i + batch_size, num_example)])
		# 返回当前批量的特征和标签
		yield features[batch_indices], labels[batch_indices]

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b):
	"""线性回归模型"""
	return torch.mamtul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
	"""均方损失"""
	return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
	"""小批量随机梯度下降"""
	with torch.no_grad(): # 上下文管理器，表示不跟踪梯度信息
		for param in params:
			param -= lr * param.grad / batch_size
			param.grad.zero_()

# 训练
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')、

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
```

- 使用sklearn实现
```python
# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成一些示例数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 在训练集上训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 可视化结果
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()
```
- 使用Pytorch简洁实现
```python
# 生成数据集
import numpy as np
import torch
from torch.utils import data

# 生成数据集的函数
def synthetic_data(w, b, num_examples):
	"""生成y=Xw+b+噪声"""
	x = torch.normal(0, 1, (num_examples, lern(w)))
	y = torch.matmul(X, w) + b
	y += torch.normal(0, 0.01, y.shape)
	return X, y.shape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 读取数据集
def load_array(data_array, batch_size, is_train=True):
	"""构造一个pytorch数据迭代器"""
	dataset = data.TensorDataset(*data_arrays)
	return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 定义模型
from torch import nn

net = nn.Sequential(nn.Linear(2, 1)) # 输入2输出1形状的全连接层

# 初始化参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 损失函数
loss = nn.MSELoss()

# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
	for X, y in data_iter:
		l = loss(net(X), y)
		trainer.zero_grad()
		l.backward()
		trainer.step()
	l = loss(net(features), labels)
	print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差', true_b - b)
```