import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy as sp
import matplotlib.pyplot as plt

n=10000
random_nos=np.random.rand(n,1)
plt.hist(random_nos,density="True",label="Uniform random numbers")
x=np.arange(0,1.1,0.1)
y=np.ones(x.size)
plt.plot(x,y,label="Uniform PDF")
plt.legend(fontsize=17)
plt.xlabel("x",fontsize=17)
plt.ylabel("PDF",fontsize=17)
plt.show()
