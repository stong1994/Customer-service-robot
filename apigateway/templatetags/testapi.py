# coding: utf-8
import numpy as np
from math import log


# 先创建数据集
def loadDataSet():
    dataList = [['我','想','要','了','解','一','下','打','个','勾'],
               ['我','购','买','了','一','些','课','程','但','是','我','找','不','到','它','们','了'],
               ['打','个','勾','是','一','个','非','常','好','的','公','司','我','想','要','了','解','更','多'],
               ['你','们','的','课','程','太','贵','了','能','给','我','打','折','吗'],
               ['我','非','常','喜','欢','打','个','勾','的','教','育','方','式','我','要','怎','么','才','能','加入','你','们'],
               ['为','什','么','我','买','了','课','程','但','是','却','不','能','观','看'],
               ['我','的','账','号','登','录','不','上'],
               ['我','注','册','了','账','号','却','不','能','登','录'],
               ['我','登','录','时','报','错','密','码','不','对']]
    dataLabel = [0,1,0,1,0,1,2,2,2]
    return dataList,dataLabel

# 获取词汇表
def createWordList(dataSet):
    wordSet = set([])
    for sentence in dataSet:
        wordSet = wordSet | set(sentence)
    return list(wordSet)

# 判断词汇表中的每个词在句子中是否出现，出现标为1，否则为0
def setOfWord2Vec(wordList, inputList):
    returnVec = [0] * len(wordList)
    for word in inputList:
        if word in wordList:
            returnVec[wordList.index(word)] += 1
        else:
            print("%s 不在词汇表中" % word)
    return returnVec

#one-vs-all 分别计算每个类别占总个数的概率和该性质语句中每个单词出现的概率
def getRate(wordMatrix, dataLabel):
    numSentence = len(wordMatrix) #数据集中句子的个数
    numWords = len(wordMatrix[0]) #词汇表的长度
    
    infoRate = sum(dataLabel) / float(numSentence) # "与课程相关"占句子个数的比例
    p1List = np.ones(numWords) # "与课程相关"性质的词 对应的词汇表
    p1WordNum = 2.0 #"与课程相关"性质的语句中 词的数量
    #遍历所有语句，分别获取两种类别性质的 词的数量和 词汇表
    for i in range(numSentence):
        if dataLabel[i] == 1:
            p1List += wordMatrix[i]
            p1WordNum += sum(wordMatrix[i])
    p1RateList = p1List / p1WordNum  #每个词是"打个勾信息"性质的词 的概率组成的列表
    #为了防止下溢出，对结果进行log()运算
    p1RateList = [log(float(p1)) for p1 in p1RateList]
    return p1RateList,infoRate

# 构建分类器：对语句进行分类
def classifySentence(wordMatrix, p0RateList, p1RateList, p2RateList, p0Rate, p1Rate, p2Rate):
    p2 = sum(wordMatrix * p2RateList) + log(p2Rate)
    p1 = sum(wordMatrix * p1RateList) + log(p1Rate)
    p0 = sum(wordMatrix * p0RateList) + log(p0Rate)
    if p1 > p0 and p1 > p2:
        return 1
    elif p2 > p1 and p2 > p0:
        return 2
    else:
        return 0

# 朴素贝叶斯分类函数
'''
1.获取数据特征和标签
2.获取词汇表
3.分别计算每个类别占总个数的概率和该性质语句中每个单词出现的概率
4.获取输入语句中每个词在词汇表中出现的次数组成的列表
5.通过输入语句中的字出现的个数和3中概率判断该语句的性质
'''
def test(sentence):
    testWordList = list(sentence)
    dataList,dataLabel = loadDataSet()
    wordList = createWordList(dataList)
    trainMatrix = [] #样本中每个句子中的单词出现在词汇表中的次数组成的矩阵
    for data in dataList:
        trainMatrix.append(setOfWord2Vec(wordList, data))
        
    #获取课程的
    #更改标签列表，"与课程相关"的标签不变，其他的为0，也就是将2变为0，以此类推，得到三种类别对应的概率
    dataLabel1 = [0 if x == 2 else x for x in dataLabel]
    p1RateList,p1Rate = getRate(np.array(trainMatrix), np.array(dataLabel1))
    dataLabel2 = [0 if x == 1 else x for x in dataLabel]
    dataLabel2 = [1 if x == 2 else x for x in dataLabel2]
    p2RateList,p2Rate = getRate(np.array(trainMatrix), np.array(dataLabel2))
    dataLabel0 = [0 if x == 2 else x for x in dataLabel]
    dataLabel0 = [1 if x == 0 else x for x in dataLabel0]
    p0RateList,p0Rate = getRate(np.array(trainMatrix), np.array(dataLabel0))

    thisDoc = np.array(setOfWord2Vec(wordList, testWordList))
    returnVal = classifySentence(thisDoc,p0RateList, p1RateList, p2RateList, p0Rate, p1Rate, p2Rate)
    print(testWordList, '被判断为',returnVal)
    return returnVal
