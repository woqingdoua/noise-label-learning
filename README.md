# Noise Label learning
=====
In the realistic environment, the labels of big data are often manually marked by experts or obtained automatically 
(For example, in the Clothing1M dataset, the clothing types are obtained by analyzing pictures' surrounding text). 
However, neither of these two methods can avoid generating wrong marks in label classification. Also, traditional 
neural network models can fit any label sample's information, which will cause the wrong (noise) label samples to
teach the model wrong knowledge. In Noise Label learning, our purpose is to reduce model fitting to what may be 
noisy samples and improve the model's Robustness. Therefore, our goal is 1. Identifying the samples that may be 
noise; 2. Reducing the weight of these samples' loss so that it does not affect the models' parameters update. 
Note that in Noise Label learning, clean samples cannot be used to train the model. In other words, all pieces
may be noise.
