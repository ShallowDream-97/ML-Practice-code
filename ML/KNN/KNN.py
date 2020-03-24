#-- coding: UTF-8 --
#knn算法的核心思想：将已经知道标签的数据分类，并在坐标系中表示，找到与未知的数据距离最近的k个数据，这k个数据决定该数据的类型
from numpy import *
import operator
from os import listdir 

def createDataSet():
	group = array ([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

def classify0(inX,dataSet,labels,k):#X作为分类的输入向量，dataSet是训练样本集，标签向量是labels，k表示邻近参照数量.这个函数是核心函数用于分类
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX,(dataSetSize,1))-dataSet#tile函数是横纵向复制数组,减法以后得到了目标与训练组中各数据的差值
	sqDiffMat = diffMat**2#平方以后差值就成了单维度上的距离
	sqDistances = sqDiffMat.sum(axis = 1)#平方和就是总距离的平方  axis=1表示按行相加 , axis=0表示按列相加
	distances = sqDistances**0.5#开方得到距离
	sortedDistIndicies = distances.argsort()#升序排列
	classCount={}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0)+1#在classCount的字典中，key值就相当于它的数目。比如第一个训练集是A，那么getA的值是0，加1，就是现在的值，再来一个就是加2.很巧妙这里。

	sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)#排序后确定分类趋向最明显的。sort与sorted函数都是排序函数，但是sort只能用于list，sorted可以用于所有迭代对象
	return sortedClassCount[0][0]

def file2matrix(filename):#这个函数是对text2训练集专门设计的数据录入函数
	fr = open (filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)#得到总行数
	returnMat = zeros((numberOfLines,3))#zeros是矩阵生成函数，它有三个参数，第一个参数是形状，用（a,b）做这个参数的意思就是生成一个A行B列的矩阵。这里就是生成一个总行数为行数，3列的矩阵
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()#strip函数用于移除字符串头尾指定的字符，默认就是空格和换行符
		listFromLine = line.split('\t')#这里是移除递进符
		returnMat[index,:] = listFromLine[0:3]#将数据储存
		classLabelVector.append(int(listFromLine[-1]))#我猜是转化为整数
		index += 1 #数量记录
	return returnMat,classLabelVector

def autoNorm(dataSet):#归一化特征值函数
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))#shape函数的功能是读取矩阵的rank
	m = dataSet.shape[0]# img.shape[0]：图像的垂直尺寸（高度）img.shape[1]：图像的水平尺寸（宽度）img.shape[2]：图像的通道数
	normDataSet = dataSet - tile(minVals,(m,1))#tile函数第二个函数参数，表示横向纵向。
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet,range,minVals#返回已经归一化的数据集，为了确保能够通过归一化后的数据找回源数据，保存了范围和最小值

def datingClassTest():
	hoRatio = 0.10#从训练集中选择出来作为测试集的比例
	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')#载入训练集，这个地方也可以是url
	normMat,ranges,minVals = autoNorm(datingDataMat)#归一化处理
	m = normMat.shape[0]#垂直尺寸
	numTestVecs = int (m*hoRatio)#从训练集中选择出来作为测试集的数目
	errorCount = 0.0#错误率计数
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print (("the classifier came back with: %d, the real answer is: %d") % (classifierResult,datingLabels[i]))
		if (classifierResult != datingLabels[i]): errorCount += 1.0
	print (("the total error rate is: %f") % (errorCount/float(numTestVecs)))
	print ((errorCount))

def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+1] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	 hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
		

