---
title: 知识表示学习Trans系列
date: 2024-05-14
author: mxm
categories:
  - 知识表示学习
tags:
  - Trans系列
mathjax: true
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
![这个图片](source/images/2024/05/14/trans_series/image1.png "向量关系")

于是我们定义一个距离$d(\vec{x}, \vec{y})$来表示两个向量之间的距离，一般情况下我们取L1或L2范数。那么我们需要对一个正确的三元组(h,r,t)来说，d(h+r, t)越小越好；错误的三元组(h,r,t)，d(h+r, t)越大越好。目标函数如下：  

$$
min \sum _{(h,r,t)} \sum _{(h^{\prime},r^{\prime},t^{\prime})}[\gamma + d(h+r,t) - d(h^{\prime}+r^{\prime},t^{\prime})]_{+}  
$$

其中$[x]_{+}=max \{0, x\}$， $\Delta^{\prime}$代表负样本，$\Delta$代表正样本  

通常为了方便训练并避免过拟合，会加上约束条件：
||h||<1, ||r||<1, ||t||<2  

论文中算法训练流程描述如下：  
![这个图片](source/images/2024/05/14/trans_series/image2.png "训练过程")


```python
class TransE(nn.Module):

    def __init__(self, entity_num, relation_num, norm=1, dim=100):
        super(TransE, self).__init__()
        self.norm = norm
        self.dim = dim
        self.entity_num = entity_num
        self.entities_emb = self._init_emb(entity_num)
        self.relations_emb = self._init_emb(relation_num)

    def _init_emb(self, num_embeddings):
        embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.dim)
        uniform_range = 6 / np.sqrt(self.dim)
        embedding.weight.data.uniform_(-uniform_range, uniform_range)
        embedding.weight.data = torch.div(embedding.weight.data, embedding.weight.data.norm(p=2, dim=1, keepdim=True))
        return embedding

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        positive_distances = self._distance(positive_triplets)
        negative_distances = self._distance(negative_triplets)
        return positive_distances, negative_distances

    def _distance(self, triplets):
        heads = self.entities_emb(triplets[:, 0])
        relations = self.relations_emb(triplets[:, 1])
        tails = self.entities_emb(triplets[:, 2])
        return (heads + relations - tails).norm(p=self.norm, dim=1)

    def link_predict(self, head, relation, tail=None, k=10):
        # h_add_r: [batch size, embed size] -> [batch size, 1, embed size] -> [batch size, entity num, embed size]
        h_add_r = self.entities_emb(head) + self.relations_emb(relation)
        h_add_r = torch.unsqueeze(h_add_r, dim=1)
        h_add_r = h_add_r.expand(h_add_r.shape[0], self.entity_num, self.dim)
        # embed_tail: [batch size, embed size] -> [batch size, entity num, embed size]
        embed_tail = self.entities_emb.weight.data.expand(h_add_r.shape[0], self.entity_num, self.dim)
        # values: [batch size, k] scores, the smaller, the better
        # indices: [batch size, k] indices of entities ranked by scores
        values, indices = torch.topk(torch.norm(h_add_r - embed_tail, dim=2), k=self.entity_num, dim=1, largest=False)
        if tail is not None:
            tail = tail.view(-1, 1)
            rank_num = torch.eq(indices, tail).nonzero().permute(1, 0)[1]+1
            rank_num[rank_num > 9] = 10000
            mrr = torch.sum(1/rank_num)
            hits_1_num = torch.sum(torch.eq(indices[:, :1], tail)).item()
            hits_3_num = torch.sum(torch.eq(indices[:, :3], tail)).item()
            hits_10_num = torch.sum(torch.eq(indices[:, :10], tail)).item()
            return mrr, hits_1_num, hits_3_num, hits_10_num     # 返回一个batchsize, mrr的和，hit@k的和
        return indices[:, :k]
    
    def evaluate(self, data_loader, dev_num=5000):
        mrr_sum = hits_1_nums = hits_3_nums = hits_10_nums = 0
        for heads, relations, tails in tqdm.tqdm(data_loader):
            mrr_sum_batch, hits_1_num, hits_3_num, hits_10_num = self.link_predict(heads.cuda(), relations.cuda(), tails.cuda())
            mrr_sum += mrr_sum_batch
            hits_1_nums += hits_1_num
            hits_3_nums += hits_3_num
            hits_10_nums += hits_10_num
        return mrr_sum/dev_num, hits_1_nums/dev_num, hits_3_nums/dev_num, hits_10_nums/dev_num


```

## TransH
论文链接：[paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.486.2800&rep=rep1&type=pdf)  
源码：[github](https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/TransH.py)  

TransH 是Zhen Wang 等人在2014年提出的一种对于TransE模型的改进方案，这个模型的具体思路是将三元组中的关系（relation, 或者 predicate），抽象成一个向量空间中的超平面（Hyperplane），每次都是将头结点或者尾节点映射到这个超平面上，再通过超平面上的平移向量计算头尾节点的差值。这样做的目的，主要是因为TransE模型在反射，一对多，多对一等关系中处理的效果并不好。举个例子，在一个对多的关系中，头结点由一个向量表示，它在一个关系中指向多个不同的节点，而这个关系也是由一个vector表示的；理想情况下，如果h + r - t = 0，这就会导致指向的多个节点所计算出来的向量都是相同的值，这明显是有问题的。而且同时，这也造成一个节点在连接不同relations中时，都使用一个相同的vector表示。而在TransH中，为了改善上述问题，引入超平面来代替原有关系向量，从而使得同一个节点在不同关系超平面的向量表示不尽相同。如下图所示：  
![这个图片](source/images/2024/05/14/trans_series/image3.png "TransE和TransH对比")

根据上图，我们可以得一个三元组元素的数学表示，h和t分别代表头结点和尾节点的向量，而关系超平面由平面的法向量 
$w_r$ 以及平面上的平移向量 $d_r$表示。具体的算法实现，对于一个三元组，我们首先需要将h和t映射到我们的超平面上，从而得到映射向量$h_\perp$和$t_\perp$, 具体公式如下：

$$
h_\perp = h - w_r^{T}*h*w_r  
$$

$$
t_\perp = t - w_r^{T}*t*w_r  
$$

其中简单说明下$w_r^{T}h w_r$的含义，这里$w_r^{T}*h=|w||h|\cos$表示h在$w_r$方向上投影的长度(带正负号)，乘以$w_r$即h在$w_r$的投影。  
得到投影之后我们就可以根据下面的score function来求得三元组的差值:  

$$
f_r(h,t) = ||h_\perp + d_r - t_\perp||_2^{2}
$$

这个公式中所期望的结果为，如果三元组关系是正确的，则结果数值较小，反之则结果数值较大。为了实现上述所期望的结果，作者引入了margin-base ranking function 作为损失函数来训练模型。

$$
min \sum _{(h,r,t)} \sum _{(h^{\prime},r^{\prime},t^{\prime})}[\gamma + f_r(h,t) - f_r(h^{\prime},t^{\prime})]_{+}  
$$

TransH与TransE还有一点不同之处，在于负例的生成。现实中的知识图谱不完整，需要减少假负例（即替换了一个节点后的三元组，恰好是整个知识图谱中存在的另一个三元组）的出现，因此需要根据头尾节点关系，进行节点替换。比如对于一对多的关系，我们更多的替换头结点而不是尾节点，这样才能避免假负例出现的情况，具体的标准如下。  
对于一个关系r, 我们首先要统计两个数值，即这个关系每个头结点平均对应的尾节点数，记做 tph；及这个关系每一个尾节点平均对应的头节点数，记做 hpt 。最后通过公式$p=\frac{tph}{tph+hpt}$来表示头结点没被替换的概率，而尾节点替换的概率为1-p。  
