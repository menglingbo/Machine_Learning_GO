from numpy import *
import operator

'''
使用近邻算法对南瓜和黄瓜分类
'''


class KNearAlgorithm:

    myGroup = array([[40, 0.4], [10, 2.0], [50, 0.6], [13, 2.5], [15, 3.0], [46, 0.8]])
    labels = ['黄瓜', '南瓜', '黄瓜', '南瓜', '南瓜', '黄瓜']

    '''用K_近邻算法分类'''
    def classify(self, newData, dataSet, labels, k):
        '''shape函数是用来检查一个矩阵或数组的维度'''
        dataSetSize = dataSet.shape[0]
        '''tile函数成一个新的矩阵'''
        diffMat = tile(newData, (dataSetSize, 1)) - dataSet
        distances = ((diffMat ** 2).sum(axis=1)) ** 0.5
        '''将distances中的元素从小到大排序并提取其索引'''
        arrIndex = distances.argsort()
        classCount = {}
        for i in range(k):
            label = labels[arrIndex[i]]
            classCount[label] = classCount.get(label, 0) + 1
        sorted_class_count = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]


x = KNearAlgorithm()
print(x.classify([18, 10], x.myGroup, x.labels, 5))
