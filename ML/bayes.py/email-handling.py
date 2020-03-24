#--coding: UTF-8 --
#这是关于使用贝叶斯分类器对电子邮件进行过滤的处理
import re
import bayes
def textParse(bigString):#定义一个规整化邮件文本函数
	listoftokens = re.split(r'\W*',bigString)#将字符串进行分割
	return [tok.lower() for tok in listoftokens if len(tok)>2] #将分割后的词汇全部转为小写，
	#并排除长度小于2的词
    
def spamTest():
	doclist = []
	classlist = []
	fulltext = []
	for i in range(1,26):#打开所有邮件样本，并汇总其中的文本及词汇
		emailText = open('D:/Anaconda/test/机器学习/Ch04/email/spam/{}.txt'.format(i),encoding='gbk').read()
		wordlist = textParse(emailText)
		doclist.append(wordlist)
		fulltext.extend(wordlist)
		classlist.append(1)
		emailText = open('D:/Anaconda/test/机器学习/Ch04/email/ham/{}.txt'.format(i),encoding='gbk').read()
		wordlist = textParse(emailText)
		doclist.append(wordlist)
		fulltext.extend(wordlist)
		classlist.append(0)
	vocablist = createVocabList(doclist)#建立词汇集
	trainingSet = list(range(50))#总文档数是50
	testSet = []
	for i in range(10):#随机抽取10个文档作为测试集
		randIndex = int(np.random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del trainingSet[randIndex]
	trainMat = []
	trainClasses = []
	for docIndex in trainingSet:#剩下的40个文档为训练集
		trainMat.append(setofwordsvec(vocablist,doclist[docIndex]))
		#将剩下40个文档转化为向量后放入trainMat列表中
		trainClasses.append(classlist[docIndex])#将剩下40个文档的对应类型放到trainClasses列表中
	p0V,p1V,pSpam = trainNBO(np.array(trainMat),np.array(trainClasses))
	#计算概率
	errorCount = 0
	for docIndex in testSet:#利用测试集中的文档验证错误率
		wordVector = setofwordsvec(vocablist,doclist[docIndex])
		if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classlist[docIndex]:
  			errorCount += 1
	print('the error rate is :',float(errorCount)/len(testSet))