from numpy import *
import operator
import matplotlib.pyplot as plt
import pickle
'''
决策树ID3算法，这里我们把特征属性里面的特征值数值化。
T: outlook-->sunny = 0, overcast = 1, rain = 2;
T: Temperature--> hot = 0, mild = 1, cool = 2;
T: Humidity--> high = 0, normal = 1;
T: Windy--> true = 1, false = 0;
其目标分类是 Play 是否去学习？
'''


class TreeId3:

    # dataSet = [[0, 0, 0, 0, 'no'], [0, 0, 0, 1, 'no'], [1, 0, 0, 0, 'yes'], [2, 1, 0, 0, 'yes'],
    #                [2, 2, 1, 0, 'yes'], [2, 2, 1, 1, 'no'], [1, 2, 1, 1, 'yes'], [0, 1, 0, 0, 'no'],
    #                [0, 2, 1, 0, 'yes'], [2, 1, 1, 0, 'yes'], [0, 1, 1, 1, 'yes'], [1, 1, 0, 1, 'yes'],
    #                [1, 0, 1, 0, 'yes'], [2, 1, 0, 1, 'no']]

    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]

    labels = ['no_sur', 'flipper']
    # labels = ['outlook', 'Temperature', 'Humidity', 'Windy']

    '''计算当前的信息熵'''
    def currentEntropy(self, data):
        currentDataLen = len(data)
        currentClassList = [example[len(data[0])-1] for example in data]
        currentClassify = {}
        entropy = 0
        for cls in currentClassList:
            if cls not in currentClassify.keys():
                currentClassify[cls] = 0
            currentClassify[cls] += 1

        for v in currentClassify.values():
            entropy = entropy - float(v/currentDataLen) * log2(float(v/currentDataLen))
        return entropy

    '''按照给定的特征划分数据集'''
    def splitDataSet(self, data, t, value):
        newDataSet = []
        firstList = [ex[t] for ex in data]
        for i in range(len(firstList)):
            if firstList[i] == value:
                reducedVec = (data[i])[:t]
                reducedVec.extend((data[i])[t + 1:])
                newDataSet.append(reducedVec)

        return newDataSet

    '''选择最好的数据集划分方式'''
    def bestSplitDataSet(self, data):
        dataLen = len(data[0])-1
        baseMessageEt = 0.0
        bestFeature = -1
        for i in range(dataLen):
            messageEt = self.currentEntropy(data)
            classList = set([ex[i] for ex in data])
            currentEt = 0.0
            for tcls in classList:
                newData = self.splitDataSet(data, i, tcls)
                currentEt = currentEt + len(newData)/len(data) * float(self.currentEntropy(newData))

            if baseMessageEt < (messageEt - currentEt):
                baseMessageEt = messageEt - currentEt
                bestFeature = i

        return bestFeature

    '''
    有时候，当我们处理了所有特征属性以后，类标签依然不是唯一的。
    也就是依然含有多种分类，这个时候我们就需要采用多数决定少数的方法来确定该叶子节点的分类
    '''
    def voteDecision(self, classifyList):
        classCount = {}
        for vote in classifyList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1

        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        print(sortedClassCount)
        return sortedClassCount[0][0]

    '''通过递归的方式创建决策树'''
    def createTree(self, data, labels):
        classList = [ex[-1] for ex in data]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(data[0]) == 1:
            return self.voteDecision(classList)
        bestFeat = self.bestSplitDataSet(data)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        del(labels[bestFeat])
        featValue = [ex[bestFeat] for ex in data]
        uniqueValue = set(featValue)
        for value in uniqueValue:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(data, bestFeat, value), subLabels)

        return myTree

    '''递归获取树的叶子节点的个数'''
    def getNumberLeafs(tree):
        numberLeafs = 0
        firstKey = list(tree.keys())[0]
        secondDict = tree[firstKey]
        for myKey in secondDict.keys():
            if isinstance(secondDict[myKey], dict):
                numberLeafs += TreeId3.getNumberLeafs(secondDict[myKey])
            else:
                numberLeafs += 1
        return numberLeafs

    '''递归获取决策树的深度'''
    def getTreeDepth(tree):
        depth = 0
        firstKey = list(tree.keys())[0]
        secondDict = tree[firstKey]
        for myKey in secondDict.keys():
            if isinstance(secondDict[myKey], dict):
                currentDepth = 1 + TreeId3.getTreeDepth(secondDict[myKey])
            else:
                currentDepth = 1
            if currentDepth > depth:
                depth = currentDepth
        return depth

    '''递归画树'''
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 注（a）
    leafNode = dict(boxstyle="round4", fc="0.8")
    arrow_args = dict(arrowstyle="<-")  # 箭头样式

    def paintTree(myTree, pPosition, nodeName):
        currentTreeLeafNum = TreeId3.getNumberLeafs(myTree)
        firstKey = list(myTree.keys())[0]
        centerP = (TreeId3.paintTree.xOff +
                   (0.5 / TreeId3.paintTree.totalW + float(currentTreeLeafNum) / 2.0 / TreeId3.paintTree.totalW),
                   TreeId3.paintTree.yOff)

        TreeId3.plotMidText(centerP, pPosition, nodeName)
        TreeId3.plotNode(firstKey, centerP, pPosition, TreeId3.decisionNode)
        secondDict = myTree[firstKey]
        TreeId3.paintTree.yOff = TreeId3.paintTree.yOff - 1.0 / TreeId3.paintTree.totalD
        print("a")
        for key in secondDict.keys():
            if isinstance(secondDict[key], dict):
                print("b")
                TreeId3.paintTree(secondDict[key], centerP, str(key))
            else:
                print("c")
                TreeId3.paintTree.xOff = TreeId3.paintTree.xOff + 1.0 / TreeId3.paintTree.totalW
                TreeId3.plotNode(secondDict[key],
                                 (TreeId3.paintTree.xOff, TreeId3.paintTree.yOff),
                                 centerP,
                                 TreeId3.leafNode)
                TreeId3.plotMidText((TreeId3.paintTree.xOff, TreeId3.paintTree.yOff),
                                    centerP,
                                    str(key))
        TreeId3.paintTree.yOff = TreeId3.paintTree.yOff + 1.0 / TreeId3.paintTree.totalD

    def plotNode(Nodename, centerPt, parentPt, nodeType):  # centerPt节点中心坐标  parentPt 起点坐标
        TreeId3.createTreePicture.axes.annotate(Nodename, xy=parentPt, xycoords='axes fraction', xytext=centerPt,
                               textcoords='axes fraction', va="center", ha="center", bbox=nodeType,
                               arrowprops=TreeId3.arrow_args)

    def plotMidText(centerP, parentP, txtString):  # 在两个节点之间的线上写上字
        xMid = (parentP[0] - centerP[0]) / 2.0 + centerP[0]
        yMid = (parentP[1] - centerP[1]) / 2.0 + centerP[1]
        TreeId3.createTreePicture.axes.text(xMid, yMid, txtString)  # text() 的使用

    '''现在我们就可以来图形化我们的节点树了'''
    def createTreePicture(myTree):
        plt.figure(1, facecolor='w')
        plt.clf()
        TreeId3.createTreePicture.axes = plt.subplot(111, frameon=True)
        TreeId3.paintTree.totalW = float(TreeId3.getNumberLeafs(myTree))  # 5
        TreeId3.paintTree.totalD = float(TreeId3.getTreeDepth(myTree))  # 2
        TreeId3.paintTree.xOff = -0.5/TreeId3.paintTree.totalW  # 0.1
        TreeId3.paintTree.yOff = 1.0
        TreeId3.paintTree(myTree, (0.5, 1.0), '')
        plt.show()

    '''使用决策树进行分类'''
    classLabel = ''

    def treeClassify(tree, labels, vec):
        firstKey = list(tree.keys())[0]
        secondDict = tree[firstKey]
        index = labels.index(firstKey)
        for key in secondDict.keys():
            if vec[index] == key:
                if isinstance(secondDict[key], dict):
                    TreeId3.treeClassify(secondDict[key], labels, vec)
                else:
                    TreeId3.classLabel = secondDict[key]

        return TreeId3.classLabel


x = TreeId3()
tree = x.createTree(x.dataSet, x.labels)
TreeId3.createTreePicture(tree)


