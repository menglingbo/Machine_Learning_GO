import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
x1 = [40., 10., 50., 13., 15., 46.]
x2 = [0.4, 2.0, 0.6, 2.5, 3.0, 0.8]
y = [1, 0, 1, 0, 0, 1]

# f(x1,x2) = m0 + m1*x1 + m2*x2 这是我们的拟合函数，
aLen = 0.00099999 # 这是梯度算法的步长
maxCount = 10000 # 这是最大迭代次数
count = 0
m0 = 0
m1 = 0
m2 = 0
while True:
    if count >= maxCount:
        break
    count += 1
    for i in range(len(x1)):
        wC = (m0 + m1*x1[i] + m2*x2[i]) - y[i]
        m0 -= aLen * wC
        m1 -= aLen * wC * x1[i]
        m2 -= aLen * wC * x2[i]

print(m0)
print(m1)
print(m2)
print(count)
plt.plot(x1, x2, 'o')
plt.plot([10, 50], [(m1*10+m0)/m2, (m1*50+m0)/m2], 'o--')
plt.ylabel('kg')
plt.xlabel('cm')
plt.show()

# 函数拟合的不好，因为训练的样本数据集太少，而且数据是我随便写的，不准确。
