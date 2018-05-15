import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('magic04.txt', sep=',', engine='python', header=None,names=(['a','b','c','d','e','f','g','h','i','j','k']))
del train['k']

data_a=train['a']
a_mean=data_a.mean()
a_std=data_a.std()

def normfun(x,mu, sigma):
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

x1 = np.arange(-150, 250,1)

# x数对应的概率密度
y1 = normfun(x1, a_mean, a_std)

# 参数,颜色，线宽
plt.plot(x1,y1, color='g',linewidth = 3)

plt.show()