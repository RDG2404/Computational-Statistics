from random import random
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

train_img=np.loadtxt(r"C:\Users\rishi\OneDrive\Desktop\Georgia Tech\Homeworks\Comp Stat\Homework3\data\data.dat", unpack=True)
label=np.loadtxt(r"C:\Users\rishi\OneDrive\Desktop\Georgia Tech\Homeworks\Comp Stat\Homework3\data\label.dat", unpack=True).reshape(-1,1)
scaler=StandardScaler()
#Fitting on dataset
scaler.fit(train_img)
#Transforming on dataset
train_img=scaler.transform(train_img)


pca=PCA(n_components=5) #creates pca with 5 components
pca.fit(train_img)
train_img=pca.transform(train_img)


def covar(n):
    x=np.random.normal(0,1,size=(n,n))
    i=np.identity(n)
    return np.dot(x, x.transpose())+i #Si*Si.Transpose+I(n) <-as given in question


print(np.random.normal(0,1,size=(1,1990)).shape)#Setting initial parameters 
def initialize_params():
    n=5 #no. of dimensions
    params= {'phi':np.random.uniform(0,1),
            'mu0':np.random.normal(0,1,size=(n,)),
            'mu1':np.random.normal(0,1,size=(n,)),
            'sigma0':covar(n),
            'sigma1':covar(n)}
    return params


#Expectation step
def e_step(x, params):
    np.log([stats.multivariate_normal(params["mu0"], params["sigma0"]).pdf(x), stats.multivariate_normal(params["mu1"], params["sigma1"]).pdf(x)])
    log_p_y_x=np.log([1-params["phi"], params["phi"]])[np.newaxis, ...] + np.log([stats.multivariate_normal(params["mu0"], params["sigma0"]).pdf(x), stats.multivariate_normal(params["mu1"], params["sigma1"]).pdf(x)]).T
    log_p_y_x_norm=logsumexp(log_p_y_x, axis=1)
    return log_p_y_x_norm, np.exp(log_p_y_x-log_p_y_x_norm[..., np.newaxis])


#Maximization step 
def m_step(x, params):
    total_count=x.shape[0]
    _,heuristics = e_step(x, params)
    heuristic0=heuristics[:,0]
    heuristic1=heuristics[:,1]
    sum_heuristic0=np.sum(heuristic0)
    sum_heuristic1=np.sum(heuristic1)
    phi=(sum_heuristic1/total_count)
    mu0=(heuristic0[...,np.newaxis].T.dot(x)/sum_heuristic0).flatten()
    mu1=(heuristic1[...,np.newaxis].T.dot(x)/sum_heuristic1).flatten()
    diff0 = x-mu0
    sigma0 = diff0.T.dot(diff0*heuristic0[...,np.newaxis])/sum_heuristic0
    diff1=x-mu1
    sigma1=diff1.T.dot(diff1*heuristic1[..., np.newaxis])/sum_heuristic1
    params={'phi':phi, 'mu0': mu0, 'mu1': mu1, 'sigma0':sigma0, 'sigma1':sigma1}
    return params


#Avg log likelihood function
def get_avg_log_likelihood(x, params):
    loglikelihood,_=e_step(x, params)
    return np.mean(loglikelihood)


#running e-m function
def run_em(x, params):
    avg_loglikelihoods=[]
    while True:
        avg_loglikelihood=get_avg_log_likelihood(x, params)
        avg_loglikelihoods.append(avg_loglikelihood) #adding log likelihoods to list in order to plot convergence
        if len(avg_loglikelihoods)>2 and abs(avg_loglikelihoods[-1]-avg_loglikelihoods[-2])<0.0001:
            break
        params=m_step(train_img, params)
        print("\tphi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
               % (params['phi'], params['mu0'], params['mu1'], params['sigma0'], params['sigma1'])) #printing the means and covariances - part b
        _, posterior=e_step(train_img,params)
        forecasts=np.argmax(posterior, axis=1)
    return params['sigma0'], params['sigma1'],forecasts, posterior, avg_loglikelihoods


init_params=initialize_params()
cov1, cov2, forecasts, posterior, loglikelihood=run_em(train_img, init_params)
print("Total Steps: ", len(loglikelihood))
plt.plot(loglikelihood)
plt.title("Log-Likelihood Plot")
plt.xlabel("Log-likelihoods")
plt.ylabel("No. of iterations")
plt.show()

#part (b)
proj=pca.inverse_transform(cov1)
#proj=pca.inverse_transform(proj)
print(proj.shape)
plt.imshow(proj, interpolation='nearest')
plt.show()

