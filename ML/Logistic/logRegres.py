#-- coding: UTF-8 --
'''逻辑回归的主要特性，在于其和纯数学考量下线性回归的差别
在线性回归下，我们对数据的考量是来源于纯数字的，最终形成的
回归模型可能符合概率论的要求，但是往往脱离了现实中的一些考
量标准。
比如，我们设计一个女性区分器，假如说想要确定一个人是女性，
需要具有如下特征：第一性特征、第二性特征、染色体型为XX如果
一个服用过量的雌性激素的男人摆在我们的分类器前，根据线性规划
的原理很有可能被判断成女性（达到前两个要求），但实际上他改变
不了他是男性的生理学事实，这个时候就要使用逻辑回归的思想，对
判断标准进行“符合逻辑”的量纲调整（与统一化标准相似，但统一化
标准是由设计者人为规定权重）
'''
'''
调整的方法，我们采用引入回归系数的概念，注意，其跟权重的概念相似但有显著的区别。将影响因素x，
转化成w*x，由此，问题转化为寻求最恰当的回归系数。
'''
'''
如何确定最佳的回归系数呢，我们使用微积分中的偏导数，寻求极大值
（或极小值）的思想，在给定任意一个单点数据的情况下，不断求偏导
，逼近最佳点。这个思路还有一个好听的名字叫做：梯度上升（梯度衰
减）。写成微分数学表达式：y2 = y1 + 德尔塔x*斜率 . d是偏导数。(1,2,3)(2,3,4)()
'''
from numpy import *

def loadDataSet():
	dataMat = []
	labelMat = []
	fr = open('testSet.txt')#打开数据文档
	for line in fr.readlines():
		lineArr = line.strip().split()#strip用于移除字符串头尾指定的字符，默认是空格和换行，spilt用于字符串切片，分成一个个单独的单词
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])#把数据添加到末尾
		labelMat.append(int(lineArr[2]))#把类别添加到末尾
	return dataMat,labelMat

def sigmoid(inX):#构造一个阶跃函数：为什么选择阶跃函数，因为从上面的情景举例，我们就能看出，我们最终拟合出的函数具有“突变性”，即达到某种条件前一定是0，过线后一定是1
	return 1.0/(1+exp(-inX))

#算法
def gradAscent(dataMatIn,classLabels):
	dataMatrix = mat(dataMatIn)#mat函数将矩阵转换类型为numpy的矩阵（单纯类型转换，因不转换无法使用相关处理函数）
	labelMat = mat(classLabels).transpose()#将矩阵转置
	m,n = shape(dataMatrix)#求矩阵的长和宽
	alpha = 0.001#计算α
	maxCycles = 500#步数最高值，为了防止过分拟合降低效率，我们只拟合最多500步（同时也规避了低效率的螺旋陷阱）
	weights = ones((n,1))#创造一个n行1列的矩阵用来存放计算出来以后的回归系数,并且将所有值设为1，因为1代表满系数，是最大的。优化就是将其调整
	#zeros ones 
	#矩阵运算，为了完成系数迭代更新。
	for k in range(maxCycles):#在最多循环次数以内，寻找最佳系数，，这个循环非常非常重要，
		h = sigmoid(dataMatrix * weights)#将两矩阵相乘，即将datamatrix每一行完全相加。
		error = (labelMat - h)#差值：和目标向量之间的偏差
		weights = weights + alpha * dataMatrix.transpose()*error#更新相关系数，
	return weights

def plotBestFit(weights):#画出数据集和Logistic回归最佳拟合直线的函数
	import matplotlib.pyplot as plt
	dataMat,labelMat = loadDataSet()
	dataArr = array(dataMat)#将数据集形成数组
	n = shape(dataArr)[0]
	xcord1 = [];ycord1 = []
	xcord2 = [];ycord2 = []
	for i in range(n):
		if int(labelMat[i])==1:
			xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
	#Gui工具图形化界面
	fig = plt.figure()#后面是绘图部分
	ax = fig.add_subplot(111) 
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	x = arange(-3.0, 3.0, 0.1)
	#最佳拟合曲线，这里设w0x0+w1x1+w2x2=0，因为0是两个分类（0和1）的分界处（Sigmoid函数），且此时x0=1
	#图中y表示x2,x表示x1
	y = (-weights[0]-weights[1]*x)/weights[2]  
	ax.plot(x, y)#作轴
	plt.xlabel('X1'); plt.ylabel('X2');#打标签
	plt.show()


