import math
from statistics import stdev
import numpy as np
import random
from scipy import stats


# data preparation
data=[(576, 3.93), (635, 3.30), (558, 2.81), (578, 3.03), (666, 3.44),
(580, 3.07), (555, 3.00), (661, 3.43), (651, 3.36), (605, 3.13),
(653, 3.12), (575, 2.74), (545, 2.76), (572, 2.88), (594, 2.96)]



# Function-1: Bootstrap Resampling of correlation coefficients
def bs_resamples(data,  b=100): # b = batch size
    bs_replicates_n=[]
    for i in range(b):
        N=15 # length of samples
        sample=random.choices(data,k=len(data)) # randomized resampling (N = length of data, in this case 15)
        scores=np.empty(shape=N) 
        gpa=np.empty(shape=N)
        for i in range(N):
            scores[i]=sample[i][0] # separating scores and gpa from tuple
            gpa[i]=sample[i][1]
        bs_replicates_n=np.append(bs_replicates_n,np.corrcoef(scores,gpa)[1][0]) # creating list of correlations from resampled data
    return bs_replicates_n



# part-a
def part_a():
    # data preparation
    scores=np.zeros(shape=len(data))
    gpa=np.zeros(shape=len(data))
    for i in range(len(data)):
        scores[i]=data[i][0]
        gpa[i]=data[i][1]
    # calculating correlation coefficients
    corr_coef=np.corrcoef(scores,gpa)[0][1]
    print(corr_coef)



# part-b
def part_b(a=0.01,B=2000): # B = Batch Size, a = alpha
    np_bs_resamples=bs_resamples(data, B) # np - non parametric, bs - bootstrap, number of batches = 2000
    ln=len(np_bs_resamples)
    np_bs_sd=stdev(np_bs_resamples) # sd - standard deviation
    np_bs_ci=[np.mean(np_bs_resamples)-stats.t.ppf(1-a, ln-1)*np_bs_sd/math.sqrt(ln), np.mean(np_bs_resamples)+stats.t.ppf(1-a, ln-1)*np_bs_sd/math.sqrt(ln)] # confidence interval at given alpha value
    print("Standard Deviation =", np_bs_sd, "\nConfidence Interval:", np_bs_ci)



# part-c
def part_c():
    n=15 # no. of samples (to be extracted from multivariate distribution)
    B=1000 # batch size (parameter for resampling function)
    a=0.01 # alpha for CI
    scores=np.zeros(shape=len(data))
    gpa=np.zeros(shape=len(data))
    for i in range(len(data)):
        scores[i]=data[i][0]
        gpa[i]=data[i][1]
    u=np.array([np.mean(scores),np.mean(gpa)]).reshape(2,1) # mean vector
    std_m=np.array([[stdev(scores),0],[0, stdev(gpa)]]) # standard deviation matrix
    corr_m=np.corrcoef(scores, gpa) # correlation matrix
    cov_m=np.dot(std_m,np.dot(corr_m, std_m)) # covariance matrix
    y=np.random.multivariate_normal(u.reshape(2,), cov_m, size=n) # generates 'n' random samples from multivariate normal distribution generated from the scores and GPA values
    p_bs_resamples=bs_resamples(tuple(y), B) # p - parametric, bs - bootstrap, number of batches = 1000
    ln=len(p_bs_resamples)
    p_bs_sd=stdev(p_bs_resamples) #sd - standard deviation
    p_bs_ci=[np.mean(p_bs_resamples)-stats.t.ppf(1-a, ln-1)*p_bs_sd/math.sqrt(ln), np.mean(p_bs_resamples)+stats.t.ppf(1-a, ln-1)*p_bs_sd/math.sqrt(ln)] # confidence interval at given alpha value
    print("Standard Deviation =", p_bs_sd,"\nConfidence Interval:", p_bs_ci)



# Function-2: Run all the parts a, b and c
def main(): 
    print("Q1)-a. Correlation Coefficient:")
    part_a()
    print("\nQ1)-b. Non-paramteric bootstrapping:")
    part_b()
    print("\nQ1)-c. Parametric Bootstrapping:")
    part_c()


if __name__=="__main__":
    main()

