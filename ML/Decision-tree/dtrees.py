#-- coding: UTF-8 --
from math import log
import operator

#多数表决函数：用于后面特征值划分选择优势种类
def majorityCnt(classList):
	classCount = {}#申请一个字典用来计数
	for vote in classList:
		if vote not in classCount.keys() : classCount[vote] = 0
		classCount[vote] += 1
	#sorted函数第一个参数是迭代对象，第二个是比较方法，第三个是比较的属性，第四个true是降序，false是逆序	
	sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = true)
	return sortedClassCount[0][0]

#创建数据，和标签集
def createDataSet():#鉴定数据集，这个是用来测试用的。
	dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
	labels = ['no surfacing','flippers']
	return dataSet,labels

#计算香农熵
def calcShannonEnt(dataSet):
	#求出list的长度，表示计算参与训练的数据量
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:#不同的种类以及他们出现的次数
		currentLabel = featVec[-1]#每一行最后一个数据代表的是标签（类别）
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0#如果当前的类别不在已有的类别字典里，那么就将当前类别加入进去
		labelCounts[currentLabel] += 1#当前类别对应的key加1
	#对于label的标签的占比，求出label的香农熵
	shannonEnt = 0.0
	for key in labelCounts:
		#通过频率计算概率（这种方式应该是仅适用于大数据情况下的，因为频率受制于样本数量，只有在符合大数定律的情况下，才可以做这种等效）
		prob = float(labelCounts[key])/numEntries
		#计算香农熵，以2为底求对数
		shannonEnt -= prob * log(prob,2)
	return shannonEnt

#按照给定特征划分数据集
def soplitDataSet(dataSet,axis,value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0])-1#求第一行有多少列的Feature，最后一列是label列嘛
	#数据集的原始信息熵
	baseEntropy = calcShannonEnt(dataSet)
	#最优的信息增益熵，和最优的Featurn编号
	bestInfoGain,bestFeature = 0.0,-1
	#初始化所有的特征值：
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]#获取feature下的所有数据
		uniqueVals = set(featList)#获取剔重后的集合，使用set对list数据进行去重
		#创建一个临时的信息熵
		newEntropy = 0.0
		#遍历某一列的value集合，计算该列的信息熵
		#遍历当前特征中的所有唯一属性值，对每个唯一属性值划分为一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和
		for value in uniqueVals:
			subDataSet = soplitDataSet(dataSet,i,value)
			#计算概率
			prob = len(subDataSet)/float(len(dataSet))
			#计算信息熵
			newEntropy += prob*calcShannonEnt(subDataSet)
		#gain[信息增益]
		#信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分地索引值
		infoGain = baseEntropy - newEntropy
		print 'infoGain=',infoGain,'bestFeature=',i,baseEntropy,newEntropy
		if(infoGain>bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def createTree(dataSet,labels):
	classList = [example[-1]for example in dataSet]
	#如果数据集的最后一列的第一个值出现的次数 = 整个集合的数量，也就说只有一个类别，就只直接返回结果就行
	#第一个停止条件：所有的类标签完全相同，则直接返回该类标签
	#count（）函数是统计括号中的值在list中出现的次数
	if classList.count(classList[0]) == len(classList):
		return classList[0]								#这一行是说，如果一行里面只有一个类，那就不用计算了，直接返回就行了。
	#如果数据集只有一列，那么最初出现label次数最多的一类，作为结果
	#第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
	if len(dataSet[0])==1:
		return majorityCnt(classList)

	#选择最优的列，得到最优列对应的label
	bestFeat = chooseBestFeatureToSplit(dataSet)
	#获取label的名称
	bestFeatLabel = labels[bestFeat]
	#初始化myTree
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])#取出最优列，下一步分类就不用这个特征了，用其他的特征了，所以把它删掉了。
	featValues = [example[bestFeat] for example in dataSet]#example函数比较有趣，它用于元素是集合类的集合，比如两个集合a,b构成了大集合A，那么A用example【1】调用的结果就是将a[1]和b[1]提取出来组成一个新的集合
	uniqueVals = set(featValues)
	for value in uniqueVals:
		#求出剩余的标签label
		subLabels = labels[:]
		#遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
		myTree[bestFeatLabel][value] = createTree(soplitDataSet(dataSet,bestFeat,value),subLabels)#实现递归
	return myTree

#以上为决策树核心算法，下面是数据处理的一些内容
def classify(inputTree,featLabels,testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	key = testVec[featIndex]
	valueOfFeat = secondDict[key]
	if isinstance(valueOfFeat, dict): 
		classLabel = classify(valueOfFeat, featLabels, testVec)
	else: classLabel = valueOfFeat
	return classLabel

def storeTree(inputTree,filename):
	import pickle
	fw = open(filename,'w')
	pickle.dump(inputTree,fw)
	fw.close()
    
def grabTree(filename):
	import pickle
	fr = open(filename)
	return pickle.load(fr)


