import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.stats import norm
import matplotlib.pyplot as plt

mu=2
sigma=2
nsteps = 10000
theta = 5
a=[]
aprime=[]


def f(x):
  if(3<x<7):
    return 1
  
  return 0

for i in range(nsteps):
  theta_prime = theta + np.random.standard_normal()
  aprime.append(theta_prime)
  r = np.random.rand()
  if(f(theta_prime)/f(theta)>r):
    theta = theta_prime
    a.append(theta)
  else:
    a.append(theta)

steps=np.arange(0,nsteps,1)
plt.figure(1) 
plt.hist(a,bins=10,density='True',label="Random no.s via MCMC")
x=np.arange(3,7.1,0.1)
y=np.full(len(x),1/4)
plt.ylabel("$\\theta$")
plt.xlabel("$step number$")
plt.plot(x,y,label="Uniform PDF")
plt.legend(fontsize=17)

plt.figure(2)
plt.subplot(2,1,1)
plt.title("Initial 100 steps")
plt.plot(steps[:100],aprime[:100],"ob",label="genetated points")
plt.plot(steps[:100],a[:100],".-r",label="selected points")
plt.ylabel("$\\theta$")
plt.xlabel("$step number$")
plt.legend()

plt.subplot(2,1,2)
plt.title("Initial 500 steps")
plt.plot(steps[:500],aprime[:500],"ob",label="genetated points")
plt.plot(steps[:500],a[:500],".-r",label="selected points")
plt.ylabel("$\\theta$")
plt.xlabel("$step number$")
plt.legend()


plt.figure(3)
plt.subplot(2,1,1)
plt.title("Initial 1000 points")
plt.plot(steps[:1000],aprime[:1000],"ob",label="genetated points")
plt.plot(steps[:1000],a[:1000],".-r",label="selected points")
plt.ylabel("$\\theta$")
plt.xlabel("$step number$")
plt.legend()

plt.subplot(2,1,2)
plt.title("Complete Chain")
plt.plot(steps,aprime,"ob",label="genetated points")
plt.plot(steps,a,".-r",label="selected points")
plt.ylabel("$\\theta$")
plt.xlabel("$step number$")
plt.legend()
plt.show()
