import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import emcee
import corner
import matplotlib.backends.backend_pdf as pf  


x,y,sigma_y=np.loadtxt("data.txt",delimiter='&', usecols=(1,2,3), unpack=True)
pp=pf.PdfPages("q10_plots.pdf")   #All plots will be stored in this file
"Check Ques10_params.txt for output parameters and there uncertainities"


def log_likelihood(theta,x,y,yerr):
  a, b ,c = theta
  model = a*x**2 + b*x +c
  sigma2 = yerr**2
  #negative ln(L)
  return 0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
  
def log_prior(theta):
  a,b,c=theta
  if -500 < a < 500 and -500 < b < 500 and -500<c<500:
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
tau = sampler.get_autocorr_time()
print("autocorrelation times for three parameters",tau)
samples = sampler.get_chain()



plt.figure(figsize=(10,10))

plt.subplot(3,1,1)
plt.plot(samples[:,:,0],alpha=0.3) 
plt.ylabel("$a$",fontsize=18)
plt.subplot(3,1,2)
plt.plot(samples[:,:, 1],alpha=0.3) 
plt.ylabel("$b$",fontsize=18)
plt.xlabel("$step number$",fontsize=18)
plt.subplot(3,1,3)
plt.plot(samples[:,:, 2],alpha=0.3) 
plt.ylabel("$c$",fontsize=18)
plt.xlabel("$step number$",fontsize=18)
plt.tight_layout()
pp.savefig()

samples = sampler.get_chain(discard=100, thin=15, flat=True)

labels=["a","b","c"]
medians = np.median(samples, axis=0)
a_true,b_true,c_true = medians
fig = corner.corner(samples,labels=labels,truths=[a_true, b_true,c_true])
pp.savefig()

params=["a","b","c"]
f=open("Ques10_params.txt",'w')
for i in range(ndim):
  mcmc=np.percentile(samples[:, i], [16, 50, 84])
  q = np.diff(mcmc)
  parameters="%s= %0.5f sigma up=%0.5f sigma down=%0.5f "%(params[i],mcmc[1],q[1],-q[0])
  f.write(parameters)
  f.write("\n")
f.close()

a=np.argsort(x)
x=x[a]
y=y[a]
sigma_y=sigma_y[a]
plt.figure(figsize=(8,8))


for i in range(200):
  k=np.random.randint(0,len(samples))
  a=samples[k,0]
  b=samples[k,1]
  c=samples[k,2]
  if(i==1):
      plt.plot(x,a*x**2+b*x+c,"gold",label="Model with randomly chosen parameters.")
      continue
  plt.plot(x,a*x**2+b*x+c,"gold")

plt.errorbar(x,y,yerr=sigma_y,fmt="o",capsize=3.5,mfc='r',ecolor='k')  
plt.plot(x,a_true*x**2+b_true*x+c_true,"k",label="Best fit model") 
plt.xlabel("$x$",fontsize=18)
plt.ylabel("$y$",fontsize=18)
plt.legend(fontsize=15)
pp.savefig()
plt.show()
pp.close()
