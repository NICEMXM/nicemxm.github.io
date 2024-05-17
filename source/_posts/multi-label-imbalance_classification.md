---
title: 多标签文本分类中长尾分布的平衡策略
date: 2024-05-16
tags:
  - 文本分类
  - 损失函数
---

论文链接：[arxiv](https://arxiv.org/pdf/2109.04712.pdf)

文章源码：[github](https://github.com/Roche/BalancedLossNLP)

## 摘要
多标签文本分类是自然语言处理中的一类经典任务，训练模型为给定文本标记上不定数目的类别标签。然而实际应用时，
各类别标签的训练数据量往往差异较大（不平衡分类问题），甚至是**长尾分布**，影响了所获得模型的效果。**重采样（Resampling）和重加权（Reweighting）常用于应对不平衡分类问题**，但由于多标签文本分类的场景下类别标签间
存在关联，现有方法会导致对高频标签的过采样。

本项工作中，我们探讨了**优化损失函数的策略**，尤其是**平衡损失函数在多标签文本分类中的应用**。基于通用数据集 (Reuters-21578，90 个标签) 和生物医学领域数据集（PubMed，18211 个标签）的多组实验，我们发现一类
分布平衡损失函数的表现整体优于常用损失函数。研究人员近期发现该类损失函数对图像识别模型的效果提升，而我
们的工作进一步证明其在自然语言处理中的有效性。


## 引言
多标签文本分类是自然语言处理（NLP）的核心任务之一，旨在为给定文本从标签库中找到多个相关标签，可应用于搜索
（Prabhu et al., 2018）和产品分类（Agrawal et al., 2013）等诸多场景。图 1 展示了通用多标签文本分类
数据集 Reuters-21578 的样例数据（Hayes and Weinstein, 1990）。
image.png 
图1 Reuters-21578 的样例数据（仅展示文章标题）标签后面的数字代表数据集中带有该标签的数据实例个数。

当标签数据存在长尾分布（不平衡分类）和标签连锁（类别共现）时，多标签文本分类会变得更加复杂（图2）。长尾分布，
指的是一小部分标签（即头部标签）有很多数据实例，而大多数标签（即尾部标签）只有很少数据实例的不平衡分类情况。
标签连锁，指的是头部标签与尾部标签共同出现导致模型对头部标签的权重倾斜。现有的 NLP 解决方案包括但不限于：
在分类中对尾部标签重采样（Estabrooks et al., 2004; Charte et al., 2015），模型初始化时将类别共现信
息纳入考虑（Kurata et al., 2016），以及将头尾部标签混合的多任务架构方案 (Yang et al., 2020) 。但这
些方案依赖于模型架构的专门设计，或不适用于长尾分布数据。

近年来，计算机视觉（CV）领域也有不少关于多标签分类的研究。其中，优化损失函数的策略已被用于多种 CV 任务，如
对象识别（Durand et al., 2019; Milletari et al., 2016）、语义分割（Ge et al., 2018）与医学影像（
Li et al., 2020a）等。平衡损失函数，如 [**Focal loss (Lin et al., 2017)**](https://arxiv.org/abs/1708.02002)、[**Class-balanced loss (Cui et al., 2019)**](https://arxiv.org/abs/1901.05555)和[**Distribution-balanced loss (Wu et al., 2020)**](https://arxiv.org/abs/2007.09654) 等，提供了针对多标签图像分类的长尾分布和标签连锁问题的解决方案。由于损失函数的调整可以独立于模型架构地灵活嵌入常见模型，NLP 中也逐步有类似的优化损失函数的策略探索（Li et al., 2020b; Cohan et al., 2020）。例如，(Li et al., 2020b) 将医学图像分割任务中的 Dice loss (Milletari et al., 2016) 引入 NLP，显著改善了多种任务的模型效果。

本项工作中，我们将一类新的平衡损失函数引入 NLP，用于多标签文本分类任务，并使用 Reuters-21578（一个通用的小型数据集）和 PubMed（一个生物医学领域的大型数据集）数据集进行了实验。对于这两个数据集，分布平衡损失函数在总指标上优于其他损失函数，并且显著改善了尾部标签的模型表现。我们认为，平衡损失函数为多标签文本分类的应用提供了一个有效策略。

## 损失函数
多标签文本分类中，[二值交叉熵（Binary Cross Entropy, BCE）](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss)
是较常用的损失函数 (Bengio et al., 2013)。原始的 BCE 容易被大量头部标签或负样本干扰。近年来，一些新的损失函数通过
调节 BCE 的权重，实现了模型训练过程的相对平衡。我们在此回顾了三类损失函数设计。

### Focal loss (FL)
通过模型对数据实例标记标签的“难易程度”为 BCE 设计权重 (Lin et al., 2017)。对于同一数据实例，相比可轻松分类（p值接近真实值）的标签，难以标记（p值远离真实值）的标签将获得比 BCE 更高的权重。

Focal loss 损失函数pytorch实现如下：  

``` bash
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    '''
    Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    '''
    def __init__(self, num_classes, alpha=1, gamma=2, weight=0.20):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss()
        self.weight = weight
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce = self.cross_entropy(inputs, targets)
        onehot_targets = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, onehot_targets.float(), reduce=False)
        pt = torch.exp(-BCE_loss)
        FL_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return self.weight * torch.mean(FL_loss) + ce


if __name__ == '__main__':
    criterion = FocalLoss(num_classes=26)

```

### Class-balanced focal loss (CB)
Clas-balanced focal loss对FL基础上进行一步改进，通过估计数据采样的有效数量，将每个标签增量训练数据的边际效用纳入考虑，在不同训练数据支持的标签间调节权重 (Cui et al., 2019)

``` bash
import numpy as np
import torch
import torch.nn.functional as F

def focal_loss(logits, labels, alpha=None, gamma=2):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bc_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class Loss(torch.nn.Module):
    def __init__(
        self,
        loss_type: str = "cross_entropy",
        beta: float = 0.999,
        fl_gamma=2,
        samples_per_class=None,
        class_balanced=False,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
        Args:
            loss_type: string. One of "focal_loss", "cross_entropy",
                "binary_cross_entropy", "softmax_binary_cross_entropy".
            beta: float. Hyperparameter for Class balanced loss.
            fl_gamma: float. Hyperparameter for Focal loss.
            samples_per_class: A python list of size [num_classes].
                Required if class_balance is True.
            class_balanced: bool. Whether to use class balanced loss.
        Returns:
            Loss instance
        """
        super(Loss, self).__init__()

        if class_balanced is True and samples_per_class is None:
            raise ValueError("samples_per_class cannot be None when class_balanced is True")

        self.loss_type = loss_type
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced

    def forward(
        self,
        logits: torch.tensor,
        labels: torch.tensor,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """

        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels_one_hot = F.one_hot(labels, num_classes).float()

        if self.class_balanced:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * num_classes
            weights = torch.tensor(weights, device=logits.device).float()

            if self.loss_type != "cross_entropy":
                weights = weights.unsqueeze(0)
                weights = weights.repeat(batch_size, 1) * labels_one_hot
                weights = weights.sum(1)
                weights = weights.unsqueeze(1)
                weights = weights.repeat(1, num_classes)
        else:
            weights = None

        if self.loss_type == "focal_loss":
            cb_loss = focal_loss(logits, labels_one_hot, alpha=weights, gamma=self.fl_gamma)
        elif self.loss_type == "cross_entropy":
            cb_loss = F.cross_entropy(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "binary_cross_entropy":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax_binary_cross_entropy":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss

def test_standard_losses():
        import torch

        from balanced_loss import Loss

        # outputs and labels
        logits = torch.tensor([[0.78, 0.1, 0.05], [0.78, 0.83, 0.05]])  # 2 batch, 3 class
        labels = torch.tensor([0, 2])  # 2 batch

        # focal loss
        focal_loss = Loss(loss_type="focal_loss")
        loss = focal_loss(logits, labels)

        self.assertAlmostEqual(loss.item(), 0.85, delta=0.01)

        # cross-entropy loss
        ce_loss = Loss(loss_type="cross_entropy")
        loss = ce_loss(logits, labels)

        self.assertAlmostEqual(loss.item(), 1.17, delta=0.01)

        # binary cross-entropy loss
        bce_loss = Loss(loss_type="binary_cross_entropy")
        loss = bce_loss(logits, labels)

        self.assertAlmostEqual(loss.item(), 0.80, delta=0.01)

def test_balanced_losses():
    import torch

    from balanced_loss import Loss

    # outputs and labels
    logits = torch.tensor([[0.78, 0.1, 0.05], [0.78, 0.83, 0.05]])  # 2 batch, 3 class
    labels = torch.tensor([0, 2])  # 2 batch

    # number of samples per class in the training dataset
    samples_per_class = [30, 100, 25]  # 30, 100, 25 samples for labels 0, 1 and 2, respectively

    # class-balanced focal loss
    focal_loss = Loss(loss_type="focal_loss", samples_per_class=samples_per_class, class_balanced=True)
    loss = focal_loss(logits, labels)

    self.assertAlmostEqual(loss.item(), 1.17, delta=0.01)

    # class-balanced cross-entropy loss
    ce_loss = Loss(loss_type="cross_entropy", samples_per_class=samples_per_class, class_balanced=True)
    loss = ce_loss(logits, labels)

    self.assertAlmostEqual(loss.item(), 1.59, delta=0.01)

    # class-balanced binary cross-entropy loss
    bce_loss = Loss(loss_type="binary_cross_entropy", samples_per_class=samples_per_class, class_balanced=True)
    loss = bce_loss(logits, labels)

    self.assertAlmostEqual(loss.item(), 1.08, delta=0.01)

if __name__ == '__main__':
    # Clas-balanced损失函数，对于Clas-balanced focal loss使用loss_type="focal_loss"
    test_balanced_losses()
    # 标准损失函数
    test_standard_losses()

```
### Distribution-balanced loss (DB)
Distribution-balanced loss（DB，分布平衡损失函数）则是在 FL 基础上添加了两部分组件 (Wu et al., 2020)。其一为 Rebalancing 组件，减少了标签连锁带来的冗余信息，其二为 Negative Tolerant Regularization （NTR）组件，在不同正负样本数目的标签间调节权重，降低尾部标签的阈值。


``` bash
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResampleLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True, partial=False,
                 loss_weight=1.0, reduction='mean',
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 focal=dict(
                     focal=True,
                     alpha=0.5,
                     gamma=2,
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 class_freq=None,
                 train_num=None):
        super(ResampleLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.alpha = focal['alpha'] # change to alpha

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        self.class_freq = torch.from_numpy(np.asarray(class_freq)).float().cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = train_num # only used to be divided by class_freq
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias ########################## bug fixed https://github.com/wutong16/DistributionBalancedLoss/issues/8

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        if self.focal:
            logpt = self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(-logpt)
            wtloss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            alpha_t = torch.where(label==1, self.alpha, 1-self.alpha)
            loss = alpha_t * ((1 - pt) ** self.gamma) * wtloss ####################### balance_param should be a tensor
            loss = reduce_loss(loss, reduction)             ############################ add reduction
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def logit_reg_functions(self, labels, logits, weight=None): 
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale  + logits * labels
            if weight is not None:
                weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if  'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    torch.sum(gt_labels, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, avg_n)).cuda()
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)).cuda() / \
                      (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
                     torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, min_n)).cuda()
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight
    

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss

if __name__ == 'main':
    # 每种类别数
    class_freq = [1775, 656, 3062, 336]
    # 总类别数
    train_num = 17623
    # DB
    loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                             class_freq=class_freq, train_num=train_num)


```

## 数据集
本项工作中，我们使用了两个不同数据量和领域的多标签文本分类数据集（表 1）。**Reuters-21578 数据集**包含1987 年刊登在路透社的一万多份新闻文章（Hayes and Weinstein, 1990）。我们按照（Yang and Liu, 1999）使用的训练-测试分割数据，并将 90 个标签平均分为头部（30 个标签，各含 ≥35 个实例）、中部（31 个标签，各含 8-35 个实例）和尾部（30 个标签，各含 ≤8 个实例）标签的子集。**PubMed 数据集**则来自 BioASQ 竞赛（Licence：8283NLM123），包含PubMed 文章的标题、摘要及对应的生物医学主题词标记 (MeSH)（Tsatsaronis et al.，2015; Coordinators, 2017）。类似地，18211个标签按分位数分为头部（6018 个标签，各含≥50 个实例）、中部（5581 个标签，各含 15-50 个实例）和尾部（6612 个标签，各含 ≤15 个实例）标签的子集。

## 实验
我们比较了不同损失函数与经典 SVM one-vs-rest 模型的表现。对于各个数据集和模型，我们计算了标签集整体以及头部、中部、尾部标签子集的micro-F1 和 macro-F1 得分（Wu et al., 2019；Lipton et al., 2014 ）。表 2 汇总了不同损失函数的实验结果。Reuters-21578 结果中，BCE 的表现最差。依次对比 micro-F1 和 macro-F1之间、及不同组间的得分可以看出长尾分布的影响。PubMed 数据由于不平衡更明显，长尾分布的影响更大。


对于 Reuters-21578 数据集，损失函数 FL、CB、R-FL 和 NTR-FL 在头部标签中的表现与 BCE 相似，但在中部和尾部标签中的表现优于 BCE，说明它们对于不平衡问题的改进。DB 在尾部标签改进最明显，整体表现也优于先前使用相同数据集的解决方案，例如 Binary Relevance、EncDec、CNN、CNN-RNN、Optimal Completion Distillation和 GNN 等（Nam et al., 2017 ; Pal et al., 2020；Tsai and Lee et al., 2020）。对于PubMed 数据集，由于BCE 中部和尾部标签已失效，我们使用 FL 作为更强的基线。其他损失函数在中部和尾部标签中的表现均优于 FL。DB 再次证明了其在整体、中部和尾部标签的良好效果。

我们进一步尝试从 DB 中去除一个组件，即移除 NTR 组件得到 R-FL、移除 Rebalancing 组件得到 NTR-FL，移除 FL 组件得到 DB-0FL，通过比较三个残缺模型探索对应三个组件的效果。如表 2 所示，对于两个数据集，移除 NTR 组件 (R-FL) 或 FL 组件 (DB-0FL) 会降低所有亚组的模型效果。移除 Rebalancing 组件 (NTR-FL) 产生相似的整体 micro-F1，但整体 macro-F1 及中部和尾部标签 F1 得分不如 DB，显示增加Rebalancing 组件的作用。最终，我们还尝试将 NTR-FL 与 CB 集成，从而得到一个全新的损失函数 CB-NTR，它在两个数据集上得到的所有 F1 值均优于 CB。CB-NTR 和 DB 间的唯一区别是使用 CB 权重替换了 Rebalancing 权重，而 DB 在中部和尾部标签中的表现优于或非常接近 CB-NTR，可能来自于通过 Rebalancing 权重处理标签连锁对模型效果的提升。

## 总结
针对多标签文本分类中的不平衡分类问题，我们研究了优化损失函数的策略，并系统比较了各种平衡损失函数的效果。我们首次**将 DB 引入 NLP**，并设计了**全新的平衡损失函数 CB-NTR**。在开放数据集 Reuters-21578（90 类标签，通用领域）和 PubMed（18211 类标签，生物医学领域）的实验表明，DB 的模型效果优于其他损失函数。这项研究证明，优化损失函数的策略可以有效解决多标签文本分类时不平衡分类的问题。该策略由于仅需调整损失函数，可以灵活兼容各种基于神经网络的模型框架，也适用于其他受到长尾分布影响的 NLP 任务。