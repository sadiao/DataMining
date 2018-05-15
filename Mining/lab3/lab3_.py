import numpy as np
import pandas as pd
from sklearn import tree

train = pd.read_csv('iris.txt', sep=',', engine='python', header=None,names=(['a','b','c','d','e']))
train['e'].value_counts() #类别计算
x = train[['a','b','c','d',]].astype('float64')
y=train['e']
print(x)

clf = tree.DecisionTreeClassifier(min_impurity_decrease=0.05,splitter='best',min_samples_leaf=5)
clf = clf.fit(x,y)

with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

def calcshan(dataSet):
    lenDataSet=len(dataSet)
    p={}
    H=0.0
    for data in dataSet:
        currentLabel=data[-1]  #获取类别标签
        if currentLabel not in p.keys():  #若字典中不存在该类别标签，即创建
            p[currentLabel]=0
        p[currentLabel]+=1    #递增类别标签的值
    for key in p:
        px=float(p[key])/float(lenDataSet)  #计算某个标签的概率
        H-=px*log(px,2)  #计算信息熵
    return H

def spiltData(dataSet,axis,value):    #dataSet为要划分的数据集,axis为给定的特征，value为给定特征的具体值
    subDataSet=[]
    for data in dataSet:
        subData=[]
        if data[axis]==value:
            subData=data[:axis]  #取出data中第0到axis-1个数进subData;
            subData.extend(data[axis+1:])  #取出data中第axis+1到最后一个数进subData;这两行代码相当于把第axis个数从数据集中剔除掉
            subDataSet.append(subData) #此处要注意expend和append的区别
    return subDataSet

def chooseBestFeature(dataSet):
    lenFeature=len(dataSet[0])-1    #计算特征维度时要把类别标签那一列去掉
    shanInit=calcshan(dataSet)      #计算原始数据集的信息熵
    feature=[]
    inValue=0.0
    bestFeature=0
    for i in range(lenFeature):
        shanCarry=0.0
        feature=[example[i] for example in dataSet]  #提取第i个特征的所有数据
        feature=set(feature)  #得到第i个特征所有的分类值，如'0'和'1'
        for feat in feature:
            subData=spiltData(dataSet,i,feat)  #先对数据集按照分类值分类
            prob=float(len(subData))/float(len(dataSet))
            shanCarry+=prob*calcshan(subData)  #计算第i个特征的信息熵
        outValue=shanInit-shanCarry  #原始数据信息熵与循环中的信息熵的差
        if (outValue>inValue):
            inValue=outValue  #将信息熵与原始熵相减后的值赋给inValue，方便下一个循环的信息熵差值与其比较
            bestFeature=i
    return bestFeature

def createTree(dataSet,label):
    classList=[example[-1] for example in dataSet]   #classList是指当前数据集的类别标签
    if classList.count(classList[0])==len(classList): #计算classList中某个类别标签的数量，若只有一类，则数量与它的数据长度相等
        return classList[0]
    if len(dataSet[0])==1:   #当处理完所有特征而类别标签还不唯一时起作用
        return majorityCnt(classList)
    featBest=chooseBestFeature(dataSet)  #选择最好的分类特征
    feature=[example[featBest] for example in dataSet]  #接下来使用该分类特征进行分类
    featValue=set(feature)  #得到该特征所有的分类值，如'0'和'1'
    newLabel=label[featBest]
    del(label[featBest])
    Tree={newLabel:{}}  #创建一个多重字典，存储决策树分类结果
    for value in featValue:
        subLabel=label[:]
        Tree[newLabel][value]=createTree(spiltData(dataSet,featBest,value),subLabel) #递归函数使得Tree不断创建分支，直到分类结束
    return Tree