#-- coding: UTF-8 --
from numpy import *
import logRegres
dataArr,LabelMat = logRegres.loadDataSet()
logRegres.gradAscent(dataArr,LabelMat)
weights = logRegres.gradAscent(dataArr,LabelMat)
logRegres.plotBestFit(weights.getA())