对于一些分类问题，类别之间的界限是模糊的，难以做到“硬性”区分类别，但是可以求得属于每个类别的概率，通过“软性”的方法来区分类别
## softmax函数
softmax函数是一种激活函数，常用于多分类问题中，将一组数值映射到一个概率分布，且概率之和为1.
$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{其中}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$$
其中$\mathbf{o}=(o_1, o_2, \ldots, o_n)$是输入向量
softmax函数的特性：
- 非负性
- 归一化
- 可导

## softmax回归
softmax回归其实是一个分类模型，使用softmax函数作为输出层处理多类别分类。softmax回归是一种单层神经网络。
![softmax回归是一种单层神经网络](https://zh-v2.d2l.ai/_images/softmaxreg.svg)

softmax回归的网络架构是一个全连接函数（softmax函数）

优化softmax函数值一般采用交叉熵损失函数
$$l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j.$$
并使用精度（accuracy）进行模型性能评估

- 关键点：
  - 似然函数
  - 交叉熵
  - 信息论

## softmax的代码实现
