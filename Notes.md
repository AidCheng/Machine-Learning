
# Modelling

## Linear Regression （基于最小二乘法 - 但矩阵不可乘时无法解决）

- 拟合的平面：
$$h_\theta (x) = \theta_0 + \theta _1 x_1 + \theta _2 x _2 \cdots$$
$$h_\theta(x)=\sum^n_{i=0}\theta_i x_i = \theta^Tx$$

- 误差 $\varepsilon$:
  - y-true value
    $$y^{(i)} = \theta^Tx^{(i)}+\varepsilon^{(i)}$$

### 正态分布 \ 高斯分布

- 误差$\varepsilon$是独立 （样本相互独立）且具有相同的分布， 并且服从均值为0， 方差为$\theta ^0$的高斯分布
- 高斯分布：多数情况下误差的浮动不会太大，但仍存在极其少数情况 ($\sigma$代表标准差)
    $$p(\varepsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(\varepsilon^{(i)})^2}{2\sigma^2})$$
  - According to the previous equation, we know that $\varepsilon^{(i)} = y ^ {(i)} - \theta^Tx^{(i)}$
  - Hence, by combining two equations, we know that
    $$p(y^{(i)}|x^{(i)};\theta) = \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})$$
    In the above equations, $p(y^{(i)}|x^{(i)}; \theta)$ means the possibility that $y^i =\theta^Tx^i$

## 似然函数

什么样的参数$\theta_0$和现有的数据$x$组合后恰好为真实值
$$L(\theta) = \prod^m_{i=1} p(y^i | x^i;\theta)$$

- 对数似然 - 将似然函数中的乘法转化成加法 
    $$\begin{align}\log L(\theta) & = \log \prod^m_{i=1}p(y^i | x^i;\theta)
    \\ & = \log \prod^m_{i=1} \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})
    \\ & = \sum^m_{i=1} \log \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}) 
    \\ & = m\log \frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2\sigma^2}\sum^m_i(y^{(i)}-\theta^Tx^{(i)})^2 \end{align}$$
    寻找上面函数中极大值点$\theta_0$，使这个参数与$x$组合后最接近真实值$y$, 即$\min(J)$, where
    $$J(\theta) = \frac 12 \sum ^m _ {i=1}(y^{(i)}-\theta^Tx^{(i)})^2$$

### 求解

x is a set of x(s), hence, x and here is treated as a matrix $\begin{bmatrix}x_1&x_2&x_3 \cdots \end{bmatrix}$，并对$J(\theta)$ 求关于$\theta$的偏导数，let $\frac{\delta J}{\delta \theta} = 0$

$$J(\theta) = \frac 12 \sum ^m _ {i=1} (h_\theta(x^i - y^i)^2) = \frac 12 (X\theta - y)^T(X\theta-y) $$

可得：
$$\theta_0 = (X^TX)^{-1}X^Ty$$

## 梯度下降

### 目标函数

$$J(\theta) = \frac 1 {2m} \sum ^m_{i=1} (y^i - h_\theta (x^i))^2$$

- Why $\frac 1 {2m}$?
  - For a single set of data $I(\theta) = \frac 12 (y^i - h_\theta (x^i))^2$
  - To fit more groups of data $J(\theta) = \frac 1m \sum ^m_{i=1} I(\theta)$ (Calculate the average loss function)

### 批量梯度下降

$$\frac {\partial J(\theta)}{\partial \theta_j} = -\frac 1m \sum^m_{i=1}(y^i - h_\theta(x^i))x^i_j$$

梯度向下移动，寻找新的$\theta^i_j$，即 $\theta'_j$

$$\begin{align}\theta'_j & = \theta_j + (- \frac{\partial J (\theta)}{\partial \theta}) \\
            & = \theta_j + \frac 1m \sum^m_{i=1}(y^i - h_\theta(x^i))x^i_j\end {align}$$

But for every sample, have to consider all the other samples to get the best solution. Hence the process would be rather slow.
$$O(n) = n^2$$

### 随机梯度下降

Only use one sample to calculate the desired parameter $\theta$
$$\theta'_j = \theta_j + (y^i - h_\theta(x^i))x^i_j$$

### 小批量梯度下降法 （使用一部分的样本进行计算）

$$\theta_j' = \theta_j - \alpha \frac 1{10} \sum^{i+9}_{k = i}(y^i - h_\theta(x^k))x^k_j$$

常见的batch样本数量为$2^4,2^5,2^6$
- Learning Rate (LR): the coefficient $\alpha$
  - Choose from the smallest until the accuracy is not tolerable

## Code implementation

### Analysis：

$$\theta'_j = \theta_j - \alpha \frac 1{10} \sum^{i+9}_{k = i}(y^i - h_\theta(x^k))x^k_j$$

- $m$: Total sample number
- $y^i$: The actual data
- $h_\theta(x^i)$: The predicted value
- $x_j^i$: The influence factor(s) in the sample
- $\alpha$: Learning rate

# Classification Problem (分类问题)
## Cross Validation

### `Scikit-Learn` Package

#### Confusion Matrix

- TP (True-positive)
- FP (True-positive)
- TN (True-negative)
- FN (False-nagative)

#### Calculation (code implementation)

```py
from sklearn.model_selection import cross val_predict

y_train_pred = cross_val_predict(method, x, y, cv = n)
```

- 评估方法 （confusion matrix）
  - `sklearn.metrics.confusion_matrix(y_true, y predict)`
  - This return a 2X2 matrix  
    - `negative class [[TN, FP]]`
    - `positive class [[FN, TP]]`
  - 完美的分类器应当主对角线不为0，其余为0

- Precision (精度) and Recall (召回率)
$$Precision = \frac{TP}{TP+FP}$$

$$recall = \frac{TP}{TP+FN}$$

```py
from sklearn.matrics import preision_score, recall_score
```

#### F1 Score

给予低值更高权重，如果precision and recall分数很高，则F1 score也很高

$$F_1 = \frac{2}{\frac{1}{precision} + \frac{1}{recall}} = 2 \times \frac{precision \times recall}{precision + recall} = \frac{TP}{TP + \frac{FN + FP}{2}}$$

```py
from sklearn.matrics import f1_score
```

#### 阈值
- 设置的阈值越高，precision越高，设置的阈值越低，recall值越高

```py
from sklean.linear_model import SGDClassifier
from sklearn.matrics import precision_recall_curve

sgd_clf = SGDClassifier(max_iter = 5, random_state = 42)

y_confidence_score = sgd_clf.decision_function([X[length_X]])
precisions, recalls, thresholds = precision_recall_curve (y_true, y_confidence_score) 
```

#### ROC 曲线 (Receiver operating characteristic)
- true positive rate (TPR)
- false positive rate (FPR)
![Example](Screenshot%202023-12-26%20at%2023.27.25.png)

好的分类器曲线偏向左上角 (measure the AUC (Area under curve))

```py
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# The data set performed in different threshold
fpr, tpr, threshold = roc_curve(t_true, y_confidence_score)
auc_score = roc_auc_score(y_true, y_confidence_score)
```
