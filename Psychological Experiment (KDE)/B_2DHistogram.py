from statistics import mean, stdev, variance
from unicodedata import name
from scipy.stats import norm
from numpy import ndim, size, sqrt
from sklearn.neighbors import KernelDensity
import scipy.stats as st
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

#reading data and separating columns
df= pd.read_csv(r'C:\Users\rishi\OneDrive\Desktop\Georgia Tech\Homeworks\Comp Stat\Homework3\data\n90pol.csv')
amyg=np.array([df.loc[:, 'amygdala']]).T[:,0]
acc=np.array([df.loc[:,'acc']]).T[:,0]
fig, ax = plt.subplots()
h=ax.hist2d(amyg,acc, bins=(11,11))
fig.colorbar(h[3],ax=ax)
fig.suptitle(' 2-D Histogram (Q2 - b)')
plt.xlabel('Amygdala Data')
plt.ylabel('ACC Data')
plt.show()