from statistics import mean, stdev, variance
from unicodedata import name
from scipy.stats import norm
from numpy import sqrt
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
orient_data=np.array([df.loc[:,'orientation']]).T[:,0]
#bandwidth constants
h_amyg=1.06*np.std(amyg)*len(amyg)**(-1/5) #bandwidth for amygdala
h_acc=1.06*np.std(acc)*len(acc)**(-1/5) #bandwidth for acc
#function for 2-D KDE
def kde_2d(amyg_data, acc_data, i):
    #defining borders 
    delta_acc=(max(acc_data)-min(acc_data))/10
    delta_amyg=(max(amyg_data)-min(amyg_data))/10
    acc_min=min(acc_data)-delta_acc
    acc_max=max(acc_data)+delta_acc
    amyg_min=min(amyg_data)-delta_amyg
    amyg_max=max(amyg_data)+delta_amyg
    #creating meshgrid
    accxx,amygyy=np.mgrid[acc_min:acc_max:100j, amyg_min:amyg_max:100j]#100 pts interpolation on each axis
    positions=np.vstack([accxx.ravel(), amygyy.ravel()])
    values=np.vstack([acc_data, amyg_data])
    kernel=st.gaussian_kde(values)
    f=np.reshape(kernel(positions).T, accxx.shape)
    #plotting kernel w/annotated contours
    fig=plt.figure(figsize=(8,8))
    ax=fig.gca()
    ax.set_xlim(acc_min, acc_max)
    ax.set_ylim(amyg_min, amyg_max)
    cfset=ax.contourf(accxx, amygyy, f, cmap='coolwarm')
    ax.imshow(np.rot90(f), cmap='coolwarm', extent=[acc_min, acc_max, amyg_min, amyg_max])
    cset=ax.contour(accxx, amygyy, f, cmap='coolwarm')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('ACC')
    ax.set_ylabel('Amygdala')
    plt.subplot(2,4,i)
    plt.title("2D KDE for Amygdala vs. ACC (orientation -"+str(i)+")")

#fitting estimated conditional distributions using kde - from part (d)
def sepdata(study_data, orient):
    a=np.array([])
    for i in range(len(orient_data)):
        if(orient_data[i]==orient):
            a = np.append(a, study_data[i]) #returns a new list with data corresponding to specific orientation values
    a = a.reshape(-1, 1)
    return a

plt.figure()

for i in range(2,6):
    amyg_sep=sepdata(amyg,i) #Splitting amygdala data to conditional distribution with orientation with value i (range is 2-5)
    acc_sep=sepdata(acc,i)  #Splitting acc data to conditional distribution with orientation with value i (range is 2-5)
    amyg_sep=np.nan_to_num(amyg_sep)
    acc_sep=np.nan_to_num(acc_sep)
    kde_2d(amyg_sep, acc_sep, i)
    #print(i, acc_sep)
# kde_2d(amyg, acc, 2)

plt.suptitle("Conditional joint distribution of Amygdala and ACC over Orientation")
plt.tight_layout()
plt.show()