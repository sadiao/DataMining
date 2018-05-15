import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


data = pd.read_csv('iris.txt', sep=',', engine='python', header=None,names=(['a','b','c','d','e']))
train=data['e']
del data['e']
DB = DBSCAN(eps=0.42, min_samples=5).fit(data)
core_samples_mask = np.zeros_like(DB.labels_, dtype=bool)
core_samples_mask[DB.core_sample_indices_] = True
labels = DB.labels_
n_clusters = len(set(labels))-(1 if -1 in labels else 0)  # 族的数目（忽略噪声）
print("clusters", n_clusters)
print("labels",DB.labels_)
print("min samples",DB.min_samples)
result=list(DB.labels_)
kk=[[],[],[]]
for i in range(n_clusters):
   for j in range(len(result)):
       if result[j]==i:
            kk[i].append(train[j])
print(kk)

