# Noise Label learning

In the realistic environment, the labels of big data are often manually marked by experts or obtained automatically (For example, in the Clothing1M dataset, the clothing types are obtained by analyzing pictures' surrounding text). However, neither of these two methods can avoid generating wrong marks in label classification. Also, traditional neural network models can fit any label sample's information, which will cause the wrong (noise) label samples to teach the model wrong knowledge. In Noise Label learning, our purpose is to reduce model fitting to what may be noisy samples and improve the model's Robustness. Therefore, our goal is 1. Identifying the samples that may be noise; 2. Reducing the weight of these samples' loss so that it does not affect the models' parameters update. Note that in Noise Label learning, clean samples cannot be used to train the model. In other words, all pieces may be noise.

## Method
这里，我们将现有方法分为三大类：1) 改写损失函数；2）双模型法；3）样本权重重分配。这三种方法并不独立，特别在最新的模型中，通常可见三种方法结合使用。这里我们不单独介绍第三种方法，因为第三种方法通常作为辅助方法，结合前两者使用。

首先，我们先介绍两个经典但年代有些久远但模型，Sainbayar et al.[1]及Forward[2]。两者方法相似，都是基础模型之后加一个 num_class*num_class大的矩阵，称为Noise transition matrix。测试阶段移除该矩阵。不同之处在于Noise transition matrix参数的更新策略。前者通过增加Noise transition matrix的weight decay作为正则化项，反向传播更新参数。后者将从预测结果的每个类中选择perfect example。perfect examplee为预测为某一类的样本集中，某个样本预测为该类的概率最大。

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\overline{\boldsymbol{x}}^{i}=\operatorname{argmax}_{\boldsymbol{x} \in X^{\prime}} \hat{p}\left(\tilde{\boldsymbol{y}}=\boldsymbol{e}^{i} \mid \boldsymbol{x}\right)" style="border:none;">

此时，这个perfect example预测的概率分布作为其noise matrix这个类的估计。

<img src="http://chart.googleapis.com/chart?cht=tx&chl= \hat{T}_{i j}=\hat{p}\left(\tilde{\boldsymbol{y}}=\boldsymbol{e}^{j} \mid \overline{\boldsymbol{x}}^{i}\right)" style="border:none;">

接下来，我们介绍最近常用于解决Noise Label Learning的方法。改写损失函数法包括：Trunc Loss[3], Yi et al.[4], PENCIL[5], DAC[6], Bi-Tempered[7], SL[8]. Trunc Loss was proposed based on the principle: Cross Entropy Loss收敛速度快，拟合能力强，但noise-robust差；MAE收敛速度慢，拟合label能力差，但noise-robust好。于是作者提出两者的结合：

<img src="http://chart.googleapis.com/chart?cht=tx&chl= \mathcal{L}_{q}\left(f(\boldsymbol{x}), \boldsymbol{e}_{j}\right)=\frac{\left(1-f_{j}(\boldsymbol{x})^{q}\right)}{q}" style="border:none;">
q->0是CCE，q=1是MAE。 

作者提出，样本损失在一定区间内则使用上式优化，超过此区间则设损失为常数，不优化，总的目标函数如下表达：

<img src="http://chart.googleapis.com/chart?cht=tx&chl= \underset{\boldsymbol{\theta}, \boldsymbol{w} \in[0,1]^{n}}{\operatorname{argmin}} \sum_{i=1}^{n} w_{i} \mathcal{L}_{q}\left(f\left(\boldsymbol{x}_{i} ; \boldsymbol{\theta}\right), y_{i}\right)-\mathcal{L}_{q}(k) \sum_{i=1}^{n} w_{i}" style="border:none;">





This project reproduce methods include: Trunc loss[1], PENCIL[2], MLNT[3], Co-teaching[4], Co-teaching_plus[5]




## References
[1] Sukhbaatar S, Bruna J, Paluri M, et al. Training convolutional networks with noisy labels[J]. arXiv preprint arXiv:1406.2080, 2014.

[2] Patrini G, Rozza A, Krishna Menon A, et al. Making deep neural networks robust to label noise: A loss correction approach[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 1944-1952.

[3] Zhang Z, Sabuncu M. Generalized cross entropy loss for training deep neural networks with noisy labels[J]. Advances in neural information processing systems, 2018, 31: 8778-8788.

[4] Tanaka D, Ikami D, Yamasaki T, et al. Joint optimization framework for learning with noisy labels[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 5552-5560.

[5] Yi K, Wu J. Probabilistic end-to-end noise correction for learning with noisy labels[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 7017-7025.

[6] Thulasidasan S, Bhattacharya T, Bilmes J, et al. Combating label noise in deep learning using abstention[J]. arXiv preprint arXiv:1905.10964, 2019.

[7] Amid E, Warmuth M K K, Anil R, et al. Robust bi-tempered logistic loss based on bregman divergences[C]//Advances in Neural Information Processing Systems. 2019: 15013-15022.

[8] Wang Y, Ma X, Chen Z, et al. Symmetric cross entropy for robust learning with noisy labels[C]//Proceedings of the IEEE International Conference on Computer Vision. 2019: 322-330.




[3] Li J, Wong Y, Zhao Q, et al. Learning to learn from noisy labeled data[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 5051-5059.

[4] Han B, Yao Q, Yu X, et al. Co-teaching: Robust training of deep neural networks with extremely noisy labels[C]//Advances in neural information processing systems. 2018: 8527-8537.

[5] Yu X, Han B, Yao J, et al. How does disagreement help generalization against label corruption?[J]. arXiv preprint arXiv:1901.04215, 2019.


