# Noise Label learning

In the realistic environment, the labels of big data are often manually marked by experts or obtained automatically (For example, in the Clothing1M dataset, the clothing types are obtained by analyzing pictures' surrounding text). However, neither of these two methods can avoid generating wrong marks in label classification. Also, traditional neural network models can fit any label sample's information, which will cause the wrong (noise) label samples to teach the model wrong knowledge. In Noise Label learning, our purpose is to reduce model fitting to what may be noisy samples and improve the model's Robustness. Therefore, our goal is 1. Identifying the samples that may be noise; 2. Reducing the weight of these samples' loss so that it does not affect the models' parameters update. Note that in Noise Label learning, clean samples cannot be used to train the model. In other words, all pieces may be noise.

## Method
Here, we divide the existing methods into three categories: 1) rewriteing the loss function; 2) dual model method; 3) re-weighting samples' loss. These three methods are not independent, especially in the latest models. In a other word,  three methods are commonly used in combination. Here we do not introduce the third method separately, because the third method is usually used as an auxiliary tool in combination with the first two.

First of all, we introduced two classic models, Sainbayar et al.[1] and Forward[2]. The two methods are similar, both adding a num_class*num_class matrix after the basic model, called the Noise transition matrix, and removing the matrix during the test phase. The difference lies in the update strategy of the Noise transition matrix parameters. The former used the weight decay of the Noise transition matrix as a regularization term to back-propagate and update parameters. The latter selected the perfect example from each category of the predicted results. The perfect example has the greatest possibility comparing to those samples which are predicted to be the same class.

$$\overline{\boldsymbol{x}}^{i}=\operatorname{argmax}_{\boldsymbol{x} \in X^{\prime}} \hat{p}\left(\tilde{\boldsymbol{y}}=\boldsymbol{e}^{i} \mid \boldsymbol{x}\right)$$

At this time, the probability distribution obtained by this perfect example was used as an unbiased estimate of its noise matrix class.

<img src="http://chart.googleapis.com/chart?cht=tx&chl= \hat{T}_{i j}=\hat{p}\left(\tilde{\boldsymbol{y}}=\boldsymbol{e}^{j} \mid \overline{\boldsymbol{x}}^{i}\right)" style="border:none;">

接下来，我们介绍最近常用于解决Noise Label Learning的方法。改写损失函数法包括：Trunc Loss[3], Yi et al.[4], PENCIL[5], DAC[6], SL[7]. Trunc Loss was proposed based on the principle: Cross Entropy Loss收敛速度快，拟合能力强，但noise-robust差；MAE收敛速度慢，拟合label能力差，但noise-robust好。于是作者提出两者的结合：

<img src="http://chart.googleapis.com/chart?cht=tx&chl= \mathcal{L}_{q}\left(f(\boldsymbol{x}), \boldsymbol{e}_{j}\right)=\frac{\left(1-f_{j}(\boldsymbol{x})^{q}\right)}{q}" style="border:none;">, q->0是CCE，q=1是MAE。 

作者提出，样本损失在一定区间内则使用上式优化，超过此区间则设损失为常数，不优化，总的目标函数如下表达：

<img src="http://chart.googleapis.com/chart?cht=tx&chl= \underset{\boldsymbol{\theta}, \boldsymbol{w} \in[0,1]^{n}}{\operatorname{argmin}} \sum_{i=1}^{n} w_{i} \mathcal{L}_{q}\left(f\left(\boldsymbol{x}_{i} ; \boldsymbol{\theta}\right), y_{i}\right)-\mathcal{L}_{q}(k) \sum_{i=1}^{n} w_{i}" style="border:none;">


PENCIL[5] 是 Yi et al.[4]的改进。两个方法都使用了KL-divergence loss 和 Entropy loss。前者为分类损失，后者为正则化项，目的在于可以 force the network to peak at only one category rather than being flat because the one-hot distribution has the smallest possible entropy value。不同之处在于一个先验概率分布的引入。Yi et al的方法中，引入一个固定的先验概率分布，which is a distribution of classes among all training data. 目地有两个：1.If the prior distribution of classes is known, then the updated labels should follow the same； 2. prevent the assignment of all labels to a single class；。但在PENCIL中，这个先验概率视为参数，可以被更新。这个先验概率于预测值的KL-divergence作为这两种方法损失中但正则化项。

DAC[6] 引入 abstention rates概念，根据abstention rates分配样本权重。样本输出维度为类别数目+1，最后一维度作为abstention rates。为避免模型倾向于遗弃样本，作者提出正则化项，当遗弃样本式此项会增加总体损失。因此，总的损失函数为：

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\mathcal{L}\left(x_{j}\right)=\left(1-p_{k+1}\right)\left(-\sum_{i=1}^{k} t_{i} \log \frac{p_{i}}{1-p_{k+1}}\right)+\alpha \log \frac{1}{1-p_{k+1}}" style="border:none;">， pk+1为abstention rate。

Symmetric Cross Entropy Learning (SL) 是一个很好理解的方法，文章提出 reverse cross entropy 和  reverse KL-divergence，证明其更具有鲁棒性。cross entropy和KL-divergence不是对称的函数，即求 p 对 q 的散度和 q 对 p 的散度并不相等。传统分类任务中，我们求 KL(q||p)作为损失函数, q为ground truth class distribution, p is the predicted distribution over labels. reverse KL-divergence 中求KL(p||q)作为损失函数. reverse cross entropy同理。

This project reproduce methods include: Trunc loss[1], PENCIL[2], MLNT[8], Co-teaching[9], Co-teaching_plus[10]




## References
[1] Sukhbaatar S, Bruna J, Paluri M, et al. Training convolutional networks with noisy labels[J]. arXiv preprint arXiv:1406.2080, 2014.

[2] Patrini G, Rozza A, Krishna Menon A, et al. Making deep neural networks robust to label noise: A loss correction approach[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 1944-1952.

[3] Zhang Z, Sabuncu M. Generalized cross entropy loss for training deep neural networks with noisy labels[J]. Advances in neural information processing systems, 2018, 31: 8778-8788.

[4] Tanaka D, Ikami D, Yamasaki T, et al. Joint optimization framework for learning with noisy labels[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 5552-5560.

[5] Yi K, Wu J. Probabilistic end-to-end noise correction for learning with noisy labels[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 7017-7025.

[6] Thulasidasan S, Bhattacharya T, Bilmes J, et al. Combating label noise in deep learning using abstention[J]. arXiv preprint arXiv:1905.10964, 2019.

[7] Wang Y, Ma X, Chen Z, et al. Symmetric cross entropy for robust learning with noisy labels[C]//Proceedings of the IEEE International Conference on Computer Vision. 2019: 322-330.

[8] Li J, Wong Y, Zhao Q, et al. Learning to learn from noisy labeled data[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 5051-5059.

[9] Han B, Yao Q, Yu X, et al. Co-teaching: Robust training of deep neural networks with extremely noisy labels[C]//Advances in neural information processing systems. 2018: 8527-8537.

[10] Yu X, Han B, Yao J, et al. How does disagreement help generalization against label corruption?[J]. arXiv preprint arXiv:1901.04215, 2019.


