---
title: 知识表示学习Trans系列
---

本文将简短梳理知识表示学习Trans系列方法，包含TransE、TransH、TransR、TransD、TransA、TransG、TranSparse以及KG2E，并且结合Github清华开源的高星代码了解一下实现过程，以便能通过代码看出他们之间的联系和区别。


## 知识图谱是什么？
知识图谱可以表示为一个**三元组(sub,rel,obj)**。举个例子：熊大的弟弟是熊二，表示成三元组是（熊大，弟弟，熊二）。**前者是主体，中间是关系，后者是客体**。主体和客体统称为实体（entity）。关系有一个属性，不可逆，也就是说主体和客体不能颠倒过来。知识图谱的集合，链接起来成为一个图（graph），每个节点是一个一个实体，每条边是一个关系，或者说是一个事实（fact），也就是有向图，主体指向客体。

简化起见表示成(h, r, t)，说明如下：  
h表示头实体  
r表示关系  
t表示尾实体  

## 知识表示是什么？
知识表示学习的前提是表示学习，那么何为表示学习？就是把图像、文本、语音等的语义信息表示为低维稠密的实体向量，即Embedding。Embedding是大家都熟知的，自从13年出现的word2vec，Embedding成为NLP任务的标配。

在这里便是将**知识图谱中的实体和关系向量化**，具体的说，我们的目标是将知识库中所有的实体、关系表示成一个低纬度的向量。(h, r, t)\rightarrow(h^{\rightarrow}, r^{\rightarrow}, t^{\rightarrow})

知识表示学习目前的一些主要方法包括以下几个：

**距离模型(Structured Embedding, SE)**  
**单层神经网络模型(Single Layer Model, SLM)**  
**能量模型(Semantic Matching Energy, SME)**  
**双线性模型**  
**张量神经网络模型(Neural Tensor Network, NTN)**  
**矩阵分解模型**  
**翻译模型**  

## TransE
论文链接：[paper](https://proceedings.neurips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)
文章源码：[github](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/kge/transe.py)


**翻译模型**从TransE开始，并衍生出一系列模型。  
TransE模型的基本思想就是把relation看做是head到tail的翻译，认为一个正确的知识三元组应该满足 h + r =(约等于) t，而错误的则不满足，通俗来讲就是头实体 embedding 加上关系 embedding 近似等于尾实embedding ，见下图，思想就是这么的简单但却高效。  
![这个图片](source\_posts\paper_image\trans_series\image1.png "向量关系")

于是我们定义一个距离d(\vec{x}, \vec{y})来表示两个向量之间的距离，一般情况下我们取L1或L2范数。那么我们需要对一个正确的三元组(h,r,t)来说，d(h+r, t)越小越好；错误的三元组(h,r,t)，d(h+r, t)越大越好。目标函数如下：  

min \sum_{(h,r,t)} \sum_{h^{\prime},r^{\prime},t^{\prime}}[\Gamma + d(h+r,t) - d(h^{\prime}+r^{\prime},t^{\prime})]_{+}  
其中[x]_{+}=max{0, x}  
\delta^{prime}代表负样本  
\delta代表正样本  

通常为了方便训练并避免过拟合，会加上约束条件：
||h||<1, ||r||<1, ||t||<2  

论文中算法训练流程描述如下：  
![这个图片](source\_posts\paper_image\trans_series\image2.png "训练过程")
