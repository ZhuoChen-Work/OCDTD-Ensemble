# OCDTD-Ensemble
In novelty detection tasks, One-Class deep Taylor decomposition (OC-DTD,https://arxiv.org/abs/1805.06230) has been proved as an efficient method to explain
the decision outcome of the One-Class SVM models. In this work we extend the OC-DTD model to outlier-ensemble models and propose a method (called "local Times" method) to detect and explain the anomalies. The bagging method is used as the baseline.

We compare the performance of the "bagging" and the "local Times" models.
Let C1 and C2 be two basic OC-SVM classifiers; Cx + Cy, x, y ∈ {1, 2} denotes the model
fused in the "bagging" method (Cx+Cx is identical to Cx); Cx × Cy, x, y ∈ {1, 2} denotes the model fused in the
"local Times" method. The following figure shows the numbers of support vectors of different classifiers.

![demo2](https://user-images.githubusercontent.com/118645613/212612749-f5dcc24d-9a61-4120-913b-ee27dfbb85f4.png)


The heatmap(https://www.sciencedirect.com/science/article/pii/S0031320316303582) is used to evaluate the explanation for the novelty. The "higher temperature area"  indicates this area is more likely to be classified as novelty.

Compared with the "bagging" method, although the "local Times" method sometimes may slightly lower down the classification accuracy,it leads to a better explanation for the anomalies.

# MNIST experiments 
In this experiment digits "2" are chosen as inliers, orther digits are regarded as outliers. The figure shows the heatmap of four inliers and four outliers. 

![MNIST_heat](https://user-images.githubusercontent.com/118645613/212557111-09ad1372-b7bc-458a-a371-81242aff5d81.png)

# MNIST-C experiments 
In this experiment normal digits are chosen as inliers, digits with corruption are regarded as outliers. The figure shows the heatmap of outliers with different type of corruptions. 

![MNIST-C](https://user-images.githubusercontent.com/118645613/212557113-760a1da0-5421-49f5-bef0-a36eebbb9cdb.png)

# MedMNIST experiments 

PathMNIST: choose "label 7" as inlier, "label 6" as outlier

![medMNIST1](https://user-images.githubusercontent.com/118645613/212564368-b87e20c7-4494-4dfd-a2d6-ab201a1cf3fc.png)


PneumoniaMNIST: choose "label 0" as inlier, "label 1" as outlier

![med_MNIST](https://user-images.githubusercontent.com/118645613/212557686-739803e1-10dd-4d92-b36c-4f3ebe02b065.png)


# MvTec experiments
In this experiment normal images are inliers, images with defects are outliers. MvTec offers the ground truth area of novelty, so we can use the cosin 
similarity (CS score) to evaluate the explanation of novelty, the higher the better.

![tile_crack](https://user-images.githubusercontent.com/118645613/212557115-01c8301c-afde-495a-b58c-9508115da1ee.png)

![leather_fold](https://user-images.githubusercontent.com/118645613/212612530-f72f53e9-b1d6-4020-9c5e-388830407e67.png)

![wood_hole](https://user-images.githubusercontent.com/118645613/212612599-9d9224cd-6f52-417c-a45d-dd5911f8b933.png)

