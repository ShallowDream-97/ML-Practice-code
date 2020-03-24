#-- coding: UTF-8 --
import KNN
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
group,labels = KNN.createDataSet()
datingDataMat,datingLabels = KNN.file2matrix('datingTestSet2.txt')

a,b = KNN.file2matrix("datingTestSet2.txt")

#fig = plt.figure()
#x = fig.add_subplot(111)#绘图函数，三个参数可以合并写成一个整数，也可以写成三个单独的参数，行列以及索引
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
#plt.show()

normMat,ranges,minVals = KNN.autoNorm(datingDataMat)
KNN.datingClassTest()