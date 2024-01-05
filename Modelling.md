
# Modelling

## Classification

- 优化
- 预测
- 决策/评价
- NP-hard （无法求最优）

## 文献查找

主题 + 关键词 (method)

## 数据查找

- awesome-public-datasets (Github)
- Kaggle

## 数据预处理

- Missing data
  - 大量缺失 （删除相关指标）
  - 其他处理方法：
    - 均值 或 众数差值
      - 定量数据（均值）/定性数据（众数）
      - 适用于对个体精度要求不大的数据
    - 牛顿插值法
      - 根据公式构造近似函数填补缺失值（适用性强）
      - 区间边缘不稳定（龙格现象）
      - 适用于只追求函数值而不关心变化的数据（热力学/地形/定位）
        - 例如：线性插值 （左右点相连取对应点）
    - 样条插值法
      - 用分段光滑的曲线进行差值（曲线连续+曲率连续）
      - 适用：精度高无突变的数据 （加工零件，水流量 etc.）

- 异常值：
  - 删除异常值并用上述方法填补
  - 查找异常：
    - $3\sigma$正态分布
      - $\mu \pm 3 \sigma$ （$\sigma$ 代表标准差）
      - 区间范围内的概率为 99.73%， 区间外的认定为异常值
      - 不适用于其他分布数据 （如排队论 - 泊松分布）
    - 箱型图
      - 寻找下四分位数 $Q_1$ 和上四分位数 $Q_3$
      - 四分位距 $IQR = Q_3 - Q_1$
      - 一般正常范围： $[Q_1 - 1.5 * IQR, Q_3 + 1.5 * IQR]$

## 如何建模？

- 论文：
  - Abstract / Restate Problem / Model Hypothesis / Symbol explain / Modelling / Solving / Analysis / Imporovements / Reference / Appendix
    - Abstract: 最重要
    - Restate: 简述题目
    - 模型假设: 说明哪些因素（原因需要合理）可以忽略，可以事半功倍 
  
- 模型建立：
  - 查阅资料，尝试复述模型(简单版本), 在根据题目改进模型
  - 模型求解：
    - ![Model](Solving%20Model.png)
    - 但是需要阐释求解过程中的object在本题中对应着什么
  - 针对不同的subquestion, 可以通过一个模型针对不同的问题进行改进

# Matlab 界面与操作

## 定义矩阵

```m
a = [1,2,3;
     3,2,1]
% Gives a 2x3 matrix
```

## 字符串和文本

- double quatation: String, normally a sentence
- single quatation: Multiple chars (e.g. DNA series)

Matlab 中的矩阵index从1开始而不是0

- char变量的添加

```m
a = 'a'
b = 'b'
ab = [a, b]
% ab = 'ab'
```