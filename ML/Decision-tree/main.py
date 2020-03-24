#-- coding: UTF-8 --
import dtrees
import treePlotter

fr = open('lenses.txt')
lenses = [inst.strip().split('\t')for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree = dtrees.createTree(lenses,lensesLabels)

treePlotter.createPlot(lensesTree)