import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy as sp
import matplotlib.pyplot as plt

    
def f(a):
  if(np.sqrt(np.dot(a,a))<=1):
    return 1
  else:
    return 0

n=100000
D=2


A=2*np.random.rand(n,D)-1  # Random no.s b/w -1 to 1 
I=np.zeros(n)
for i in range(n):
  I[i]=f(A[i,:])

I=sum(I)/n
print("Required integral",(2**D)*I,"Actual value within 4 decimal value %0.4f"%(np.pi))


D=10
A=2*np.random.rand(n,D)-1
I=np.zeros(n)

for i in range(n):
  I[i]=f(A[i,:])
  
I=sum(I)/n
print("Volume of 10-D sphere",(2**D)*I," and actual value within 4 decimal value %0.4f"%(np.pi**5/(120)))
