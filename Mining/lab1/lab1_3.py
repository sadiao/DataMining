import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('magic04.txt', sep=',', engine='python', header=None,names=(['a','b','c','d','e','f','g','h','i','j','k']))
del train['k']

