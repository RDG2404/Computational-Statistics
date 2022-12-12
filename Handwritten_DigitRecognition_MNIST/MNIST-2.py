from random import random
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

train_img=np.loadtxt(r"C:\Users\rishi\OneDrive\Desktop\Georgia Tech\Homeworks\Comp Stat\Homework3\data\data.dat", unpack=True)
#part (c)
scaler=StandardScaler()
#Fitting on dataset
scaler.fit(train_img)
#Transforming on dataset
train_img=scaler.transform(train_img)


pca=PCA(n_components=5) #creates pca with 5 components
pca.fit(train_img)
train_img=pca.transform(train_img)

kmeans = KMeans(n_clusters=2)
kmeans.fit(train_img)
y_kmeans=kmeans.predict(train_img)
plt.scatter(train_img[:,0], train_img[:, 1], c=y_kmeans, s=15, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', alpha=0.5)
plt.show()