import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy as sp
import matplotlib.pyplot as plt


def f(x):
  return(np.sqrt(2/np.pi)*np.exp(-x*x*0.5))
  
def g(x):
  return(1.5*np.exp(-x))
  
n=100000
x=np.random.rand(n)
x=-np.log(x)
y=np.random.rand(n)*g(x)

x=x[y < f(x)]
z=np.arange(0,5,0.01)

plt.hist(x,bins=50,density='True',label="Random no.s via Rejection method")
plt.plot(z,f(z),label="$\\sqrt{\\frac{2}{\\pi}}e^{-\\frac{x^2}{2}}$")
plt.xlabel("x",fontsize=17)
plt.ylabel("PDF",fontsize=17)
plt.legend(fontsize=17)
plt.show()
