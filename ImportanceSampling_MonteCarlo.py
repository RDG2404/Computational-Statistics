from cmath import pi
import math
from statistics import variance
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Function for original function f(x)
def f_x(x, mu=1, sigma=1):
    return 1/(sigma*(np.sqrt(2*pi))) * np.exp(-0.5*math.pow((x-mu)/sigma,2)) # formula for gaussian distribution 


# Function for generating Gaussian distributions
def distribution(mu, sigma):
    distribution=stats.norm(mu, sigma)
    return distribution


# Initializing given values
n=100 # Number of replications
N=500 # Number of repeats
mu_0=1 
sigma_0=1
mu_1=10.5 # mu_1=z=m*mu_0, where m is a large number
sigma_1=1

# Generating distributions of original function h(x) and new function g(x)
h_x=distribution(mu_0, sigma_0)
g_x=distribution(mu_1, sigma_1)

# Part-1: Monte Carlo Sampling
value_list_MonteCarlo=[]
value_list_ImpSampling_NRep=[]
s=[]
for j in range(N):
    for i in  range(n):
        x_i=np.random.normal(mu_0, sigma_0)
        s.append(f_x(x_i))
    value_list_MonteCarlo.append(np.mean(s)) # generating list of means of Monte Carlo samples 

    # Part-2: Importance Sampling
    value_list_ImpSampling = []
    m=mu_1/mu_0
    for i in range(n):
        x_i = np.random.normal(mu_1, sigma_1) # generating random gaussian variables from new distribution 
        v=mu_0*(m-1)*(x_i-(mu_0/2)*(m+1)) 
        imp_ratio=np.exp(v) # importance ratio formula
        value = f_x(x_i)*imp_ratio
        value_list_ImpSampling.append(value) # adding the expectation values over 100 replications 
    value_list_ImpSampling_NRep.append(np.mean(value_list_ImpSampling)) # adding the means of each 100-replication batch


print("Monte Carlo method: Variance I1 =",variance(value_list_MonteCarlo))
print("Importance Sampling method: Variance I2 =", variance(value_list_ImpSampling_NRep))