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



df= pd.read_csv(r'C:\Users\rishi\OneDrive\Desktop\Georgia Tech\Homeworks\Comp Stat\Homework3\data\n90pol.csv')
#Separating columns 
# amyg_data=np.array(df['amygdala']).reshape(-1, 1)
# acc_data=np.array(df['acc']).reshape(-1, 1)
# orient_data=np.array(df['orientation']).reshape(-1, 1)

amyg_data=np.array([df.loc[:, 'amygdala']]).T[:,0]
acc_data=np.array([df.loc[:, 'acc']]).T[:,0]
orient_data=np.array([df.loc[:, 'orientation']]).T[:,0]

#fitting estimated conditional distributions using kde - part (d)
def sepdata(study_data, orient):
    a=np.array([])
    for i in range(len(orient_data)):
        if(orient_data[i]==orient):
            a = np.append(a, study_data[i]) #returns a new list with data corresponding to specific orientation values
    a = a.reshape(-1, 1)
    return a

#bandwidth constants
h_amyg=1.06*np.std(amyg_data)*len(amyg_data)**(-1/5) #bandwidth for amygdala
h_acc=1.06*np.std(acc_data)*len(acc_data)**(-1/5) #bandwidth for acc

#fitting kde's
def kerneldensity(study_data, h, str,i):
    kde = KernelDensity(bandwidth=h, kernel='gaussian')
    kde.fit(study_data)
    study_data_d=np.linspace(np.min(study_data), np.max(study_data), 2000)[:, np.newaxis]
    log_dens=kde.score_samples(study_data_d)
    #print(study_data_d)
    plt.subplot(2,4,i)
    plt.fill(study_data_d, np.exp(log_dens), alpha=0.2, c='blue')
    plt.plot(study_data, np.full_like(study_data, -0.1), '|k', markeredgewidth=1)
    plt.title(str)
def kerneldensity2(study_data, h):
    sns.kdeplot(study_data, bw=h)
    plt.xlabel("amygdala volume")
    plt.ylabel("probability density")
    plt.show()
plt.figure()
#amygdala conditional dist.
amyg_condmean=[]
acc_condmean=[]
for i in range(2,6):
     data=sepdata(amyg_data,i) #Splitting amygdala data to conditional distribution with orientation with value i (range is 2-5)
     amyg_condmean.append(np.mean(data))
     kerneldensity(data,h_amyg,"Amygdala CD Orientation "+str(i),i-1)
#acc conditional dist.
for i in range(2,6):
     data=sepdata(acc_data,i) #Splitting amygdala data to conditional distribution with orientation with value i (range is 2-5)
     acc_condmean.append(np.mean(data))
     kerneldensity(data,h_acc,"ACC CD Orientation "+str(i),i+3)
#kerneldensity(amyg_data,h_amyg, "Amygdala CD Orientation Main", 1)
plt.suptitle("Conditional Distribution over Orientation")
plt.tight_layout()
plt.show()
print(amyg_condmean)
print(acc_condmean)