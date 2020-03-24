#-- coding: UTF-8 --
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth",fc = '0.8')#创建字典。 boxstyle=”sawtooth” 表示 注解框的边缘是波浪线，fc=”0.8” 是颜色深度。
leafNode = dict(boxstyle = "round4",fc="0.8")
arrow_args = dict(arrowstyle = "<-") #箭头样式

def plotNode(nodeTxt,centerPt,parentPt,nodeType):#节点内容，中心节点，父节点，节点格式
	createPlot.ax1.annotate(nodeTxt,xy = parentPt,xycoords = 'axes fraction',xytext = centerPt,textcoords = 'axes fraction',va = "center",ha = "center",bbox = nodeType,arrowprops = arrow_args)
## 添加注释
# 第一个参数是注释的内容
# xy设置箭头尖的坐标
# xytext设置注释内容显示的起始位置
# arrowprops 用来设置箭头
# facecolor 设置箭头的颜色
# headlength 箭头的头的长度
# headwidth 箭头的宽度
# width 箭身的宽

def createPlot():
	fig = plt.figure(1,facecolor = 'white')#创建一个新图形
	fig.clf()#清空绘图区
	createPlot.ax1 = plt.subplot(111,frameon = False)#创建画板
	plotNode('decision Node',(0.5,0.1),(0.1,0.5),decisionNode)#这两个函数调用上面定义的注释函数添加注释
	plotNode('leaf Node',(0.8,0.1),(0.3,0.8),leafNode)
	plt.show()

def getNumLeafs(myTree):
	numLeafs = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			numLeafs += getNumLeafs(secondDict[key])
		else : numLeafs +=1
	return numLeafs

def getTreeDepth(mytree):
	maxDepth = 0 
	thisDepth = 0
	firstStr = list(mytree.keys())[0]
	secondDict = mytree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			thisDepth = 1 + getTreeDepth(secondDict[key])#递归调用
		else : thisDepth =1
		if thisDepth > maxDepth: maxDepth = thisDepth

	return maxDepth

def plotMidText(cntrPt, parentPt, txtString):   #  在两个节点之间的线上写上字
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString)  # text() 的使用
def plotTree(myTree, parentPt, nodeName):  # 画树
	numleafs = getNumLeafs(myTree)
	depth = getTreeDepth(myTree)
	firstStr = list(myTree.keys())[0]
	cntrPt = (plotTree.xOff+(0.5/plotTree.totalw+float(numleafs)/2.0/plotTree.totalw), plotTree.yOff)
	plotMidText(cntrPt, parentPt, nodeName) 
	plotNode(firstStr, cntrPt, parentPt, decisionNode)
	secondDict = myTree[firstStr]
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD # 减少y的值，将树的总深度平分，每次减少移动一点(向下，因为树是自顶向下画的）
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			plotTree(secondDict[key], cntrPt, str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalw
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):  # 使用的主函数
	fig = plt.figure(1, facecolor='white')
	fig.clf()  # 清空绘图区
	axprops = dict(xticks=[], yticks=[]) # 创建字典 存储=====有疑问？？？=====
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) #  ===参数的意义？===
	plotTree.totalw = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))  # 创建两个全局变量存储树的宽度和深度
	plotTree.xOff = -0.5/plotTree.totalw # 追踪已经绘制的节点位置 初始值为 将总宽度平分 在取第一个的一半 
	plotTree.yOff = 1.0
	plotTree(inTree, (0.5,1.0), '')  # 调用函数，并指出根节点源坐标
	plt.show()


