# Noise Label learning

In the realistic environment, the labels of big data are often manually marked by experts or obtained automatically 
(For example, in the Clothing1M dataset, the clothing types are obtained by analyzing pictures' surrounding text). 
However, neither of these two methods can avoid generating wrong marks in label classification. Also, traditional 
neural network models can fit any label sample's information, which will cause the wrong (noise) label samples to
teach the model wrong knowledge. In Noise Label learning, our purpose is to reduce model fitting to what may be 
noisy samples and improve the model's Robustness. Therefore, our goal is 1. Identifying the samples that may be 
noise; 2. Reducing the weight of these samples' loss so that it does not affect the models' parameters update. 
Note that in Noise Label learning, clean samples cannot be used to train the model. In other words, all pieces
may be noise.

## Method
This project reproduce methods include: Trunc_loss[1], PENCIL[2], MLNT[3], Co-teaching[4], Co-teaching_plus[5]

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" style="border:none;">

$$y(n)=(f\ast g)(n)=\sum_{\tau =\infty}^{\infty}f(\tau )g(n-\tau )d\tau $$


## References
[1] Zhang Z, Sabuncu M. Generalized cross entropy loss for training deep neural networks with noisy labels[J]. Advances in neural information processing systems, 2018, 31: 8778-8788.

[2] Yi K, Wu J. Probabilistic end-to-end noise correction for learning with noisy labels[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 7017-7025.

[3] Li J, Wong Y, Zhao Q, et al. Learning to learn from noisy labeled data[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 5051-5059.

[4] Han B, Yao Q, Yu X, et al. Co-teaching: Robust training of deep neural networks with extremely noisy labels[C]//Advances in neural information processing systems. 2018: 8527-8537.

[5] Yu X, Han B, Yao J, et al. How does disagreement help generalization against label corruption?[J]. arXiv preprint arXiv:1901.04215, 2019.

[6] Tanaka D, Ikami D, Yamasaki T, et al. Joint optimization framework for learning with noisy labels[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 5552-5560.
