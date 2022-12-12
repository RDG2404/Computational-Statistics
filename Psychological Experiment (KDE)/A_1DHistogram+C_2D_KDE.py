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
df= pd.read_csv(r'C:\Users\rishi\OneDrive\Desktop\Georgia Tech\Homeworks\Comp Stat\Homework3\data\n90pol.csv')
#Separating columns 
amyg_data=df['amygdala']
acc_data=df['acc']
orient_data=df['orientation']
#histogram
#hist_amyg_1=plt.hist(amyg_data, bins=25) #histogram of amygdala data
#plt.show()

#kde - part (a)
h_acc=1.06*np.std(acc_data)*len(acc_data)**(-1/5) #bandwidth for acc
kde = KernelDensity(bandwidth=h_acc, kernel='gaussian')
kde.fit(acc_data[:, None])
acc_data_d=np.linspace(np.min(acc_data), np.max(acc_data), 2000)[:, np.newaxis]
log_dens=kde.score_samples(acc_data_d)
plt.fill(acc_data_d, np.exp(log_dens), alpha=0.2, c='cyan')
plt.plot(acc_data, np.full_like(acc_data, -0.1), '|k', markeredgewidth=1)
plt.xlabel("ACC Volume Data")
plt.ylabel("Probability Density")
plt.show()

# #2D histogram - part (b)
# acc_bins=np.linspace(np.min(acc_data), np.max(acc_data))
# amyg_bins=np.linspace(np.min(amyg_data), np.max(amyg_data))
# h=plt.hist2d(amyg_data,acc_data, bins=20)
# plt.colorbar(h[3])
# #plt.show()

#2D KDE - part (c)
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
plt.title('2D KDE for Amygdala vs. ACC')
plt.show()

# #fitting estimated conditional distributions using kde - part (d)
# def sepdata(study_data, orient):
#     a=[0 for i in range(len(study_data))]
#     for i in range(len(orient_data)):
#         if(orient_data[i]==orient):
#             a[i]=study_data[i] #returns a new list with data corresponding to specific orientation values
#     return a
# #initializing few values for later
# #separated datasets
# amyg2=sepdata(amyg_data, 2)
# amyg3=sepdata(amyg_data, 3)
# amyg4=sepdata(amyg_data, 4)
# amyg5=sepdata(amyg_data, 5)
# acc2=sepdata(acc_data, 2)
# acc3=sepdata(acc_data, 3)
# acc4=sepdata(acc_data, 4)
# acc5=sepdata(acc_data, 5)
# #bandwidth constants, h_acc already defined in part (a)
# h_amyg=1.06*np.std(amyg_data)*len(amyg_data)**(-1/5) #bandwidth for amygdala
# #fitting kde's
# def kerneldensity(study_data, h, str):
#     kde = KernelDensity(bandwidth=h, kernel='gaussian')
#     kde.fit(study_data[:, None])
#     study_data_d=np.linspace(np.min(study_data), np.max(study_data), 2000)[:, np.newaxis]
#     log_dens=kde.score_samples(study_data_d)
#     plt.fill(study_data_d, np.exp(log_dens), alpha=0.2, c='cyan')
#     plt.plot(study_data, np.full_like(study_data, -0.1), '|k', markeredgewidth=1)
#     plt.xlabel(str)
#     #plt.show()
# kerneldensity(acc_data,h_acc, "acc_data orient") 