# Noise Label learning

In the realistic environment, the labels of big data are often manually marked by experts or obtained automatically (For example, in the Clothing1M dataset, the clothing types are obtained by analyzing pictures' surrounding text). However, neither of these two methods can avoid generating wrong marks in label classification. Also, traditional neural network models can fit any label sample's information, which will cause the wrong (noise) label samples to teach the model wrong knowledge. In Noise Label learning, our purpose is to reduce model fitting to what may be noisy samples and improve the model's Robustness. Therefore, our goal is 1. Identifying the samples that may be noise; 2. Reducing the weight of these samples' loss so that it does not affect the models' parameters update. Note that in Noise Label learning, clean samples cannot be used to train the model. In other words, all pieces may be noise.

## Method
这里，我们将现有方法分为三大类：1) 改写损失函数；2）双模型法；3）样本权重重分配。这三种方法并不独立，特别在最新的模型中，通常可见三种方法结合使用。

首先，我们先介绍两个经典但年代有些久远但模型，Sainbayar et al.[1]及Forward[2]。两者方法相似，都是基础模型之后加一个 num_class*num_class大的矩阵，称为Noise transition matrix。测试阶段移除该矩阵。不同之处在于Noise transition matrix参数的更新策略。前者通过增加Noise transition matrix的weight decay作为正则化项，后者将从预测结果的每个类中选择perfect example，即某个样本有属于某类的最大的概率，此时，这个perfect example预测的概率分布作为其noise matrix这个类的估计。

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\overline{\boldsymbol{x}}^{i}=\operatorname{argmax}_{\boldsymbol{x} \in X^{\prime}} \hat{p}\left(\tilde{\boldsymbol{y}}=\boldsymbol{e}^{i} \mid \boldsymbol{x}\right)" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl= \hat{T}_{i j}=\hat{p}\left(\tilde{\boldsymbol{y}}=\boldsymbol{e}^{j} \mid \overline{\boldsymbol{x}}^{i}\right)" style="border:none;">


<script type="text/javascript"
   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$
\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)

This project reproduce methods include: Trunc_loss[1], PENCIL[2], MLNT[3], Co-teaching[4], Co-teaching_plus[5]




## References
[1] Sukhbaatar S, Bruna J, Paluri M, et al. Training convolutional networks with noisy labels[J]. arXiv preprint arXiv:1406.2080, 2014.

[2] Patrini G, Rozza A, Krishna Menon A, et al. Making deep neural networks robust to label noise: A loss correction approach[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 1944-1952.



[1] Zhang Z, Sabuncu M. Generalized cross entropy loss for training deep neural networks with noisy labels[J]. Advances in neural information processing systems, 2018, 31: 8778-8788.

[2] Yi K, Wu J. Probabilistic end-to-end noise correction for learning with noisy labels[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 7017-7025.

[3] Li J, Wong Y, Zhao Q, et al. Learning to learn from noisy labeled data[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 5051-5059.

[4] Han B, Yao Q, Yu X, et al. Co-teaching: Robust training of deep neural networks with extremely noisy labels[C]//Advances in neural information processing systems. 2018: 8527-8537.

[5] Yu X, Han B, Yao J, et al. How does disagreement help generalization against label corruption?[J]. arXiv preprint arXiv:1901.04215, 2019.

[6] Tanaka D, Ikami D, Yamasaki T, et al. Joint optimization framework for learning with noisy labels[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 5552-5560.
