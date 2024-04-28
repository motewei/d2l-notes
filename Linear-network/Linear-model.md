## 1. 线性模型（Linear Model）

**线性模型**就是用线性函数来解决预测问题的模型。线性函数是通过特征和输出标签之间的线性关系建立的。核心思想是通过对输入特征的线性组合来预测输出。
一般情况下，线性函数表示为：$$f(\pmb{x})=w_1x_1+w_2x_2+...+w_dx_d+b$$
向量形式为：$$f(\pmb{x})=\pmb{w}^T\pmb{x}+b$$
其中：
- $x_1, x_2, \dots, x_n$ 是输入特征
- $w_1, w_2, \dots, w_n$ 是权重 weight ，表示每个特征对输出的影响程度
- $b$ 是偏差 bias 。表示模型的截距或偏差，在没有输入特征时作为模型的输出值。  

## 2. 常见的线性模型

1. [[线性回归|线性回归（Linear Regression）]]
2. [[logistic regression]]
3. 多分类学习
4. 线性判别分析（Linear Discriminant Analysis）
5. [[感知机]]