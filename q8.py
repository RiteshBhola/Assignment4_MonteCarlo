import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy as sp
import matplotlib.pyplot as plt

    
def f(a):
  if(np.dot(a,a)<=1):
    return 1
  else:
    return 0

n=100000
D=2


A=np.random.rand(n,D)  # Random no.s b/w 0 to 1 as integral needs to be calculated in one quadrant. 
I=np.zeros(n)
for i in range(n):
  I[i]=f(A[i,:])

I=sum(I)/n
print("Required integral",(2**D)*I,"Actual value within 4 decimal value %0.4f"%(np.pi))

"Volume of 10-D sphere"
D=10
A=np.random.rand(n,D)
I=np.zeros(n)

for i in range(n):
  I[i]=f(A[i,:])
  
I=sum(I)/n
print("Volume of 10-D sphere",(2**D)*I," and actual value within 4 decimal value %0.4f"%(np.pi**5/(120)))
