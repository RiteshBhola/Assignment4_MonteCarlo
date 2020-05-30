import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import emcee
import corner


x,y,sigma_y=np.loadtxt("data.txt",delimiter='&', usecols=(1,2,3), unpack=True)
#print(x,y,yerr)

def log_likelihood(theta,x,y,yerr):
  a, b ,c = theta
  model = a*x**2 + b*x +c
  sigma2 = yerr**2
  #negative ln(L)
  return 0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
  
def log_prior(theta):
  a,b,c=theta
  if -500 < a < 500 and -500 < b < 500 and 0<c<1000:
    return 0.0
  return -np.inf
  
def log_probability(theta, x, y, yerr):
  lp = log_prior(theta)
  if not np.isfinite(lp):
    return -np.inf
  return lp - log_likelihood(theta, x, y, yerr)
  
guess = (1.0, 1.0,1.0)
soln = minimize(log_likelihood,guess,args=(x, y, sigma_y))

nwalkers, ndim = 50, 3
pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability,args=(x, y, sigma_y))
sampler.run_mcmc(pos, 4000)

samples = sampler.get_chain()
plt.subplot(2,1,1)
plt.plot(samples[:, :, 0],"k") 
plt.ylabel("$a$",fontsize=18)



plt.subplot(2,1,2)
plt.plot(samples[:, :, 1],"k") 
plt.ylabel("$b$",fontsize=18)
plt.xlabel("$step number$",fontsize=18)
plt.figure()
plt.plot(samples[:, :, 2],"k") 
plt.ylabel("$c$",fontsize=18)
plt.xlabel("$step number$",fontsize=18)
samples = sampler.get_chain(flat='True')

labels=["a","b","c"]
medians = np.median(samples, axis=0)
a_true,b_true,c_true = medians
fig = corner.corner(samples,labels=labels,truths=[a_true, b_true,c_true])
plt.savefig("plotq10_3.png",dpi=1000)
da_16=np.percentile(samples[:,0],16)
da_84=np.percentile(samples[:,0],84)
db_16=np.percentile(samples[:,1],16)
db_84=np.percentile(samples[:,1],84)
dc_16=np.percentile(samples[:,2],16)
dc_84=np.percentile(samples[:,2],84)
print("The value of parameters are:")
print("a=",a_true," 16 percentile= ",da_16," 84 percentile= ",da_84)
print("b=",b_true," 16 percentile= ",db_16," 84 percentile= ",db_84)
print("c=",c_true," 16 percentile= ",dc_16," 84 percentile= ",dc_84)
a=np.argsort(x)
x=x[a]
y=y[a]
sigma_y=sigma_y[a]
plt.figure()
plt.errorbar(x,y,yerr=sigma_y,fmt=".b")

for i in range(200):
  k=np.random.randint(0,nwalkers*4000)
  a=samples[k,0]
  b=samples[k,1]
  c=samples[k,2]
  if(i==1):
      plt.plot(x,a*x**2+b*x+c,"yellow",label="Model with randomly chosen parameters.")
      continue
  plt.plot(x,a*x**2+b*x+c,"yellow")
  
plt.plot(x,a_true*x**2+b_true*x+c_true,"k",label="Best fit model") 
plt.xlabel("$x$",fontsize=18)
plt.ylabel("$y$",fontsize=18)
plt.legend(fontsize=18)
plt.show()
