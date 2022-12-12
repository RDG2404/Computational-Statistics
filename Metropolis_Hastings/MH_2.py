import statistics
from scipy.stats import norm
import numpy as np
import pandas as pd
import random
from scipy import rand
import matplotlib.pyplot as plt
from scipy import stats

# Define the data
theta_mesh=np.linspace(0,1,11)
x=theta_mesh[random.randrange(len(theta_mesh))]
x=57
n = 100

# Define the Likelihood P(x|p) - binomial distribution
def likelihood(p):
    return stats.binom.pmf(x, n, p)

def prior(p):
    return stats.norm.pdf(p)

# Create function to compute acceptance ratio
# This function will accept the current and proposed values of p
def acceptance_ratio(p, p_new):
    # Return R, using the functions we created before
    return min(1, ((likelihood(p_new) / likelihood(p)) * (prior(p_new) / prior(p))))

# Create empty list to store samples
results = []

# Initialzie a value of p
p = np.random.normal(0, 1)

# Define model parameters
n_samples = 5000
burn_in = int(n_samples*0.2)
lag = 5

# Create the MCMC loop
for i in range(n_samples):
    # Propose a new value of p randomly from a uniform distribution between 0 and 1
    p_new = np.random.random_sample()
    # Compute acceptance probability
    R = acceptance_ratio(p, p_new)
    # Draw random sample to compare R to
    u = np.random.random_sample()
    # If R is greater than u, accept the new value of p (set p = p_new)
    if u < R:
        p = p_new
    # Record values after burn in - how often determined by lag
    if i > burn_in and i%lag == 0:
        results.append(p)

mean=np.mean(results)
print(mean)
#xb = np.repeat(results, 2)[:-1] # trace
plt.hist(results, bins=int(np.sqrt(n_samples)))
#plt.plot(xb)
# Plot between -10 and 10 with .001 steps.
# x_axis = np.linspace(0,1,11)
  
# # Calculating mean and standard deviation
# mean = statistics.mean(x_axis)
# sd = statistics.stdev(x_axis)
  
# plt.plot(x_axis, norm.pdf(x_axis, mean, sd))
plt.show()
# plt.plot(stats.uniform.pdf(0,1))
# plt.show()