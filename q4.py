import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy as sp
import matplotlib.pyplot as plt

a=np.loadtxt("q4_plot_data.dat",usecols=0)
b=np.loadtxt("q4_plot_data.dat",usecols=1)
x=np.arange(0,5,0.1)
pdf=2*np.exp(-2*x)

plt.subplot(1,2,2)
plt.hist(b,bins=100,density='True',label="Random numbers via Transformation method")
plt.plot(x,pdf,label="PDF:2$e^{-2y}$")
plt.legend(fontsize=17)
plt.xlabel("x",fontsize=17)
plt.ylabel("PDF",fontsize=17)


plt.subplot(1,2,1)
plt.hist(a,bins=10,density='True',label="Uniform distribution")
plt.xlabel("x",fontsize=17)
plt.ylabel("PDF",fontsize=17)
plt.show()
