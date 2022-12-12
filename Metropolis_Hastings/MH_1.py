import numpy as np
import pandas as pd
import random
from scipy import rand
import matplotlib.pyplot as plt
from scipy import stats

def MH_sampling(T=5000, S=5, n=10):
    theta_mesh=np.linspace(0,1,11)
    sigma=0.1
    # random initialization
    x=np.zeros(T)
    #print(x)
    x[0]=theta_mesh[random.randrange(len(theta_mesh))]

    for i in range(T):
        q = np.exp(-(theta_mesh - x[i])**2/2/sigma**2)
        q=q/sum(q)
        q_sum=np.cumsum(q)
        theta=min(abs(q_sum-random.uniform(0,1))) # proposed transition
        coin=random.uniform(0,1) # uniform random number for reject/accept
        p = (theta**S/x[i]**S)*((1-theta)**(n-S)/(1-x[i])**(n-S))*(np.cos(4*np.pi*theta))**2/np.cos(4*np.pi*x[i])**2 # acceptance ratio
        p=min(1,p)

        if coin<p:
            x[i]=theta
        else:
            x[i]=x[i-1]
    xb = np.repeat(x, 2)[:-1]
    print(np.mean(x))
    #plt.hist(x, bins=int(np.sqrt(T)))
    #plt.plot(xb)
    #print(x)
    
MH_sampling()

T=1000
n=100
p0=0.1
S=np.random.binomial(np.random.uniform(0,1), p0)

# theoretical distribution
theta_mesh=np.linspace(0,1,11)
thry_theta = 2*theta_mesh**S*(1-theta_mesh)**(n-S)*(np.cos(4*np.pi*theta_mesh))**2
thry_theta = thry_theta/sum(thry_theta)

# plt.plot(theta_mesh, thry_theta)
# plt.plot(S)
# plt.show()