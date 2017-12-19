# coding=utf-8
import operator
from os import listdir

import matplotlib.pyplot as plt
from numpy import *


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 用于分类的输入向量inX,  训练样本集合dataSet, 标签向量labels(其元素数目与dataset的行数相同), k 是选择k个邻近的k
def classify0(inX, dataSet, labels, k):
    """分类器"""
    # 此部分使用欧式距离公式求出距离
    dataSetSize = dataSet.shape[0]  # shape() 用来求矩阵的维数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 将数据集中的每个点都与待分分类的点相减,tile(a,b)表示把a重复b次数
    sqDiffMat = diffMat ** 2  # 平方差
    sqDistance = sqDiffMat.sum(axis=1)  # 平方差的和  (axis = 1 表示沿着行的方向相加，反之是沿着列方向相加)
    distances = sqDistance ** 0.5  # 标准的距离公式

    sortedDistIndicies = distances.argsort()  # 返回从小到大排序的索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 这句话的意思是对cassCount中的值进行+1操作，如果没有出现过就初始化为1
        # dict.get(k,d) get相当于一条if...else...语句,参数k在字典中，
        # 字典将返回dict[k]也就是k对应的value值；如果参数k不在字典中则返回参数d。

    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 按照逆序进行排序, itemgetter() 用来获取对象的某一个域的值,这里面表示的是用字典（可以想象为一个二维数组），其中的第二维度的值
    # 也就是每一个'A'.'B'出现的次数作为关键字来进行排序
    return sortedClassCount[0][0]


def file2matrix(filename):
    """将数据文件转换成需要的格式,将特征与标签分离"""
    fr = open(filename)
    arrayOLines = fr.readlines()  # readlne()比readlines() 慢得多，前者是一次读取整个文件，而后者仅仅是读了一行.
    numberOfLines = len(arrayOLines)  # 获取文件行数 len() 用于返回对象的长度过着元素的个数
    returnMat = zeros((numberOfLines, 3))  # zeros() 用于创建给定类型的矩阵并将其初始化为0,此处是用来创建与文件行数相同的有三列属性的矩阵
    classLabelVector = []  # 用来存储所有可能的标签
    index = 0
    for line in arrayOLines:
        line = line.strip()  # strip()用于移除字符串头部和尾部指定的字符，默认为空格.
        listFromLine = line.split('\t')  # split()可以加上参数num表示分割几次
        returnMat[index, :] = listFromLine[0:3]  # returnMat[index, :]表示选取地第index行的所有列
        classLabelVector.append(int(listFromLine[-1]))  # 文本数据中最后一列给出的是约会对象的属性，所以这里面直接获取最后一个数据
        # 这两行的作用是将文本数据中的特征与标签分开存储,也是整个函数的用处
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """用于将数字特征值转换成0~1区间的值"""
    minVals = dataSet.min(0)  # 参数为空表示求所有的最小值，为1每列的最小值，为2每行的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 使用公式 newvalue = (oldvalus-min)/(max-min)
    normDataSet = dataSet - tile(minVals, (m, 1))  # 表示行m行，每列进重复1次（若为2则会1,2,3,1,2,3这样的重复两次）
    normDataSet /= tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    """分类器效果检测"""
    hoRatio = 0.10  # 取10%数据来检测分类器的效果
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVal = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back  with %d, the real answer is : %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("total is %f " % (errorCount / float(numTestVecs)))


def img2vector(filename):
    """将32×32像素的图片转换成1*1024的向量"""
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handWritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # listdir()函数可以获取每个目录下面的所有文件的名字
    m = len(trainingFileList)
    traningMat = zeros((m, 1024))  # 需要的训练矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 获取文件名
        # 下面两条语句为什么不写成一条直接使用一split('+')应该就足以使用了吧
        fileStr = fileNameStr.split('.')[0]  # 获取不带扩展的文件名字
        classNumStr = fileStr.split('_')[0]  # 获取标签
        hwLabels.append(int(classNumStr))
        traningMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    # 这里为什么不重新写一个函数呢？？？？这样结构不是会更加清晰一点嘛
    testFileList = listdir('testDigits')
    errorCount = 0.0  # 个人觉得此处使用float是为了提高错误率的精度
    mTest = len(testFileList)
    testMat = zeros((m, 1024))
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, traningMat, hwLabels, 3)
        print("the classifier came back  with %d, the real answer is : %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("total is %f " % (errorCount / float(mTest)))


handWritingClassTest()
# datingClassTest()
# group, labels = createDataSet()
# print classify0([0, 0], group, labels, 3)
# datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
# fig = plt.figure()  # 用于新建画布
# ax = fig.add_subplot(111)  # 子图，在一张图里面画多个图,111表示将画布分成一行一列，本图放在第一个分块上
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
# 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1])
# 绘制散点图,详细参数见 http://blog.csdn.net/qiu931110/article/details/68130199
# plt.show()
# normMat, ranges, minVal = autoNorm(datingDataMat)
# print(normMat)
