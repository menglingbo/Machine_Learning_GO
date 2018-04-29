

class Bayes:
    listStr = ['玩的真他妈菜垃圾', '小哥哥玩的真好爱你哟', '原来帅哥是个大神啊', '这他妈还能玩吗傻逼',
               '今天我就日了狗了遇到你们这些傻逼', '大神带我装逼带我飞啊',
               '遇到一个百年打野的大傻逼草尼玛的', '帅哥你玩的这么好能带带小妹吗',
               '这操作6的飞起啊', '相信我能把你们带飞的']

    classes = [1, 0, 0, 1, 1, 0, 1, 0, 0, 0]

    str1: str = '我曹尼你妈'  # ？

    '''
    这是一个训练函数
    其实训练过程的本质就是在不断的丰富我们的词袋向量集和分类词向量集
    '''
    classifyDict = {'a': []}

    def getWordsVco(self, document, classify):
        #  1.首先要创建一个词袋向量,a定义为词袋向量集
        self.classifyDict['a'] = set(set(document).union(self.classifyDict['a']))
        # 2.根据分类对词向量进行归类
        if classify not in self.classifyDict.keys():
            self.classifyDict[classify] = list(document)
        else:
            self.classifyDict[classify] = self.classifyDict[classify] + list(document)

    '''判断文档分类'''
    def judgeClassify(self, inputDocument, wordsClassifyVco, classes):
        setClassify = set(classes)
        documentList = list(inputDocument)
        maxP = 0  # 最大概率
        bestClass = -1
        for cls in setClassify:
            firstGL = classes.count(cls)/len(classes)  # 定义的先验概率
            # featDict = {}
            likeValue = 1  # 定义似然值
            for featValue in documentList:
                likeValue = likeValue * ((wordsClassifyVco[cls].count(featValue)+1)/(len(wordsClassifyVco[cls])+len(wordsClassifyVco['a'])))

            classifyP = likeValue * firstGL
            if classifyP > maxP:
                maxP = classifyP
                bestClass = cls

        return bestClass


x = Bayes()
for i in range(len(x.listStr)):
    x.getWordsVco(x.listStr[i], x.classes[i])

print(x.judgeClassify(x.str1, x.classifyDict, x.classes))




