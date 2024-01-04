![这是图片](source/_posts/paper_image/mind/image1.png "llm模型发展历程")
[llm模型发展历程](https://mp.weixin.qq.com/s/-HWYRFdk1rwrKzYa0k5wZQ)
# 1、chatgpt
- chatgpt 和 mindshow结合，利用markdown语法快速生成ppt。
- chatgpt提示词：参考网友总结-[chatgpt shotcut](https://www.aishort.top/?tags=favorite)
# 2、文本相似度进展
- [知乎](https://www.zhihu.com/question/22004262)
# 3、PaLM模型
- [类chatgpt的github实现](https://github.com/lucidrains/PaLM-rlhf-pytorch)
# 4、LLaMA模型
- [论文](https://arxiv.org/pdf/2302.13971.pdf)
- 权重：[hugging face](https://huggingface.co/decapoda-research)
- 中文增量训练LLaMa模型：[github](https://github.com/CVI-SZU/Linly)
# 5、chatllama模型
- [github](https://github.com/juncongmoo/chatllama)
# 6、其它
预训练已经成为自然语言处理任务的重要组成部分，为大量自然语言处理任务带来了显著提升。

6.1 UER-py（Universal Encoder Representations）是一个用于对通用语料进行预训练并对下游任务进行微调的工具包。UER-py遵循模块化的设计原则。通过模块的组合，用户能迅速精准的复现已有的预训练模型，并利用已有的接口进一步开发更多的预训练模型。通过UER-py，我们建立了一个模型仓库，其中包含不同性质的预训练模型（例如基于不同语料、编码器、目标任务）。用户可以根据具体任务的要求，从中选择合适的预训练模型使用。
- [github](https://github.com/dbiir/UER-py/wiki/%E4%B8%BB%E9%A1%B5)


6.2 [Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft)

# 7、图文生成

[CLIP](https://zhuanlan.zhihu.com/p/493489688)

[styleClip](https://github.com/orpatashnik/StyleCLIP)

[Image Mixer](https://huggingface.co/spaces/lambdalabs/image-mixer-demo)

7.1 stable diffusion

[stable-diffusion-web使用教程1](https://mp.weixin.qq.com/s?__biz=Mzg5ODkxMDIxMw==&mid=2247484040&idx=1&sn=58d81026ade26b46943aa2bd247a1502&chksm=c05a13e2f72d9af47c50df141a60cdcd50adc8a211c1ac60eb2d07f68aa564501c29daffd5fd&scene=21#wechat_redirect) 

[stable-diffusion-web使用教程2](https://mp.weixin.qq.com/s/M4SY8XtWqRm1qdzBrNSZkQ)

[指令微调stable-diffusion](https://huggingface.co/blog/instruction-tuning-sd)

https://stable-diffusion-art.com/how-stable-diffusion-work/

[Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models](https://yuval-alaluf.github.io/Attend-and-Excite/)

# 8、图文检索
CLIP 预训练模型：https://github.com/haofanwang/natural-language-joint-query-search

https://zhuanlan.zhihu.com/p/626748008

https://zhuanlan.zhihu.com/p/619120794

# 9、fastapi
https://fastapi.tiangolo.com/zh/


# 10、LangChain

LangChain+embedding构建知识问答系统： https://zhuanlan.zhihu.com/p/641132245

# 11、pix2struct

[论文](https://arxiv.org/pdf/2210.03347.pdf)

# 12、deploy

[论文](https://arxiv.org/pdf/2212.10505.pdf)

训练过程

第一个阶段：初始化MATCHAT架构和权重

第二个阶段：微调matchat，表格数据被表示成markdown格式，训练语料：

https://aclanthology.org/2022.acl-long.277/

https://github.com/vis-nlp/chart-to-text

kaggle比赛：https://www.kaggle.com/competitions/benetech-making-graphs-accessible/overview  


# 13、优化器

1) adafactor

2) lion  


两者对比：https://zhuanlan.zhihu.com/p/609462814

# python

python：https://python3-cookbook.readthedocs.io/zh_CN/latest/preface.html  

# 14、推理

1、vllm：https://github.com/vllm-project/vllm

2、inference中的sharding

# 15、生成站点文档

https://docusaurus.io/docs

https://github.com/outerbounds/nbdoc

# 16、TorchScale

高效训练transformer

Magneto

# 17、API性能测试工具

1、jmeter

2、webbench

# 19、agent

引入SOP：https://github.com/aiwaves-cn/agents

# 20、kaggle比赛

1）LLM语言模型-科学问题检索：[Kaggle - LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446422)

2）机器学习发展专题报告比赛-生成式ai:[2023 Kaggle AI Report](https://www.kaggle.com/code/trushk/2023-kaggle-ai-report-generative-ai)

3）stabe diffusion prompt:[Stable Diffusion - Image to Prompts](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts)

# 21、path-dropout

1) Flip(fast language image pre-training)

[paper](https://arxiv.org/pdf/2212.00794.pdf)

# 22、多模态

1）[kosmos](https://arxiv.org/pdf/2302.14045.pdf)

* 数据集，模型输入格式



* 模型架构

优化点，相比传统transform

（1）MAGNETO，经典transform变体

（2）xpos, 位置编码

