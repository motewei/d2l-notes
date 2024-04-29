## 1. 什么是梯度下降法
**梯度下降法（Gradient Descent）** 是一种求最优解的迭代优化算法。基本思想是**通过沿着目标函数梯度（或负梯度）的方向逐步调整参数，减小目标函数的值。**  
![](https://raw.githubusercontent.com/motewei/DrawingBed/main/img/Pasted%20image%2020230820163607.png)
  
梯度下降法是许多机器学习算法的基础，包括[[线性回归]]、[[logistic regression]]、[[神经网络]]等。要注意避免算法陷入局部最小值或发散。
## 2. 梯度下降法的基本步骤
- 选择初始值$\mathbf w_0$
- 计算梯度（即计算目标函数的偏导数）
- 更新参数：沿梯度方向减去**梯度乘以学习率**。学习率$\eta$：步长的超参数
- 重复迭代参数，使得接近最优解。迭代法则：   $$\mathbf{w}_t=\mathbf{w}_{t-1}-\eta\frac{\partial \ell }{\partial\mathbf{w}_{t-1}}$$
## 3. 梯度下降法的两种主要类型
- 批量梯度下降：每次迭代使用整个训练数据来计算目标函数的梯度，需要大量的计算资源。可以少量迭代收敛。
- 随机梯度下降：每次迭代随机选择一个小样本 mini-batch 来计算目标函数梯度，但是由于随机小批量，噪音较多，收敛会加快，需要多次迭代收敛。

## 4. 实际使用的MSGD(minibatch stochastic gradient descent)
(1) 初始化模型参数的值
(2) 从数据集中随机抽取小批量样本且在负梯度方向上更新参数，不断迭代
$$
\begin{split}\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}\end{split}
$$
$\eta$是learning rate， $\mathcal{B}$是batch size。  
