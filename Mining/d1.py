import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normfun(x,meam,sigma):
    pdf=np.exp(-((x-mean)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return pdf

train = pd.read_csv('magic04.txt', sep=',', engine='python', header=None,names=(['a','b','c','d','e','f','g','h','i','j','k']))
del train['k']
print(train.cov())
#print(trainT.cov())
print(train.corrwith(train['a']))
trainT=train.T
print(train.mean())

x=train['a']
mean=train['a'].mean()
var=train['a'].var()
#y=train['b']
y=normfun(x,mean,var)
#plt.scatter(x,y)
plt.plot(x,y)

plt.show()

