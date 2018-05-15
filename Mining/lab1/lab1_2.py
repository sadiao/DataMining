import numpy as np
import pandas as pd

train = pd.read_csv('magic04.txt', sep=',', engine='python', header=None,names=(['a','b','c','d','e','f','g','h','i','j','k']))
del train['k']
#中心化
center=train-train.mean()
#转置
centert=center.T

n=train['a'].count()
#协方差等于
cov=centert.dot(center)/n
print(cov)