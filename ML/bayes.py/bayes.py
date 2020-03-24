#-- coding: UTF-8 --
#朴素贝叶斯的思想，来自于利用条件概率（贝叶斯公式）来比较一个数据点来自各分类的可能性，并将数据点归于概率较大的那一方。
"""朴素贝叶斯的“朴素”源自于两个理想假设：
第一假设：假设所有样本数据互相独立。这个假设显然是不符合现
实的，比如我们进行自然语言学习，那么 am 出现在 i 的频率显
然要比 am 出现在 u 之后高，也就是说，其显然是被其他因素影
响的.
第二假设：假设所有特征同等重要。每个特征的重要性显然是不同的，
比如，在这样一个情景下：你在哔哩哔哩上寻找网课的时候，影响你选
择最重要的因素是网课的质量，而不是老师的性别。
但是在朴素贝叶斯里是假定老师的性别和上课质量同等重要，以便于操作。
"""

"""朴素贝叶斯的基本过程：
1、收集数据：任何方法。
2、准备数据：需要处理成数值型和布尔型
3、分析数据：有大量数据时，绘制特征效果不好（同KNN进行比较），此时使用直方图表示好
4、训练算法：计算不同的独立特征的条件概率
5、测试算法：计算错误率
6、使用算法：可以在任意场景使用。
"""
#测试数据提供函数
def loadDataSet():
	postingList = [['my','dog','has','flea','problems','help','please'],
	['maybe','not', 'take', 'him' ,'to' ,'dog', 'park', 'stupid'],
	['my','dalmation','is','so','cute','I','love','him'],
	['stop',' posting', 'stupid', 'worthless', 'garbage'],
	['mr','licks','ate','my','steak','how','to','stop','him'],
	['quit','buying','worthless','dog','food','stupid']]
	classVec = [0,1,0,1,0,1]#1 代表侮辱性语言 0 代表正常言论
	return postingList,classVec
#创建一个无重复的词汇表
def createVocabList(dataSet):
	vocabSet = set([])#set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
	for document in dataSet:
		vocabSet = vocabSet | set(document)#|是并集运算符，表示两个集合的并集
	return list(vocabSet)
#生成词汇向量(比对词汇表，有这个词就是1，没有就是0，比如词汇表（A,B,C）向量a = （1，0，0）,意思就是，a里面有A，没有BC)
def setOfWords2Vec(vocabList,inputSet):
	returnVec = [0]*len(vocabList)#向量维度显然是和词汇表维度相同的。
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)]=1
		else:print ("the word : %s is not in my vocabulary") % word
	return returnVec

#以上为数据处理函数
#下面为贝叶斯训练函数
"""利用p（c|w）*p(w)= *p(c)p(w|c)
得到p（c|w） = ……，它的意义就是，利用某类别具有某特征的概率反推具有某特征就是某类别的概率
比如，利用雨雨的男朋友的身高是175概率，来推算一个175的男生有多大的可能性是雨雨男朋友，如果
这个概率较大，我们就认为他是雨雨的男朋友
2/3,1/3,0,2/9，1/100,

"""
def trainNBO(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)#获取有多少条文本
	numwords = len(trainMatrix[0])#获取单个文本的长度
	pAbusive = sum(trainCategory)/float(numTrainDocs)#目标类别的概率，这里因为使用的样例是用0和1区分类别，计算1的总概率只需要加和就行。
	p0Num = ones(numWords)#根据文本长度设定一个全1的向量
	p1Num = ones(numWords)#这里注意，生成的类型是np.ndarray,是用于存放同类型元素的多维数组
	p0Denom = 2.0
	p1Denom = 2.0
	for i in range(numTrainDocs):#遍历所有文本
		if trainCategory[i]==1:#如果对应的标签是1，即敏感文本
			p1Num += trainMatrix[i]#统计文本中所有单词出现的次数
		#因为类型是np.ndarry,所以这里对应位置的值是直接相加的
			p1Denom += np.sum(trainMatrix[i])#这里统计共有多少词
		#因为此例只有两个特征，即0或1，所以if以后可以直接else，否则需要多判断
		else:
			p0Num += trainMatrix[i]
			p0Denom += np.sum(trainMatrix[i])
	p1Vect = log(p1Num/p1Denom) #这里计算敏感文本中，每个词占该类型下所有词的比例，即p(wi|c1),,
	p0Vect = log(p0Num/p0Denom) #这是p（wi|c0）#i和0是下标
	#这里之所以要用log函数处理，是用了似然处理（参见数理统计似然估计的相关内容）
	return p0Vect,p1Vect,pAbusive


#这是一个分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)#这里是贝叶斯公式，之所以乘号变成加号了，是因为这是在对数里的运算
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClas1)
	if p1>p0:
		return 1
	else:
		return 0
#定义一个测试函数:通过测试两组样例，来检验是否有效
def testingNB():
	list0Posts,listClasses = loadDataSet()#载入测试数据
	myVocabList = createVocabList(list0Posts)#生成无重复的词汇表
	trainMat = []#这个列表的元素是向量
	for postinDoc in list0Posts:
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))#计算每一句语料的向量，并且添加到列表中去
	p0V,p1V,pAb = trainNBO(np.array(trainMat),np.array(listClasses))#计算贝叶斯公式计算所需要的三个概率
	testEntry = ['love','my','dalmation']#测试用小组
	thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))#测试用向量
	print(testEntry,"cassified as :",classifyNB(thisDoc,p0V,p1V,pAb))#输出测试结果
	testEntry = ['stupid','garbage']
	thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
	print(testEntry,'classified as',classifyNB(thisDoc,p0V,p1V,pAb))#输出第二组测试结果










