import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.stats import norm
import matplotlib.pyplot as plt

n=10000
x1=np.random.rand(n)
x2=np.random.rand(n)
y1=np.sqrt(-2*np.log(x1))*np.cos(2*np.pi*x2)
y2=np.sqrt(-2*np.log(x1))*np.cos(2*np.pi*x2)


mu=0
sigma=1
x=np.arange(-5,5,0.01)
y=norm(mu,sigma)

plt.hist(y1,bins=50,density="True",label="Box Muller Method")
plt.plot(x,y.pdf(x),label="Gaussian PDF")
plt.legend()
plt.xlabel("x",fontsize=17)
plt.ylabel("PDF",fontsize=17)
plt.show()
