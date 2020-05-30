import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy as sp
import matplotlib.pyplot as plt
import timeit
N=10000
my_setup="""
import numpy as np
n=10000
"""
stmt1="""
a=1664525
c=1013904223
m=4295647895
x=15

random_nos=[]

for i in range(n):
  x=(a*x+c)%m
  random_nos.append(x/m)
"""
stmt2="""
random_nos=np.random.rand(n,1)
"""
t1=timeit.timeit(setup=my_setup,stmt=stmt1,number=N)
t1/=N
t2=timeit.timeit(setup=my_setup,stmt=stmt2,number=N)
t2/=N

print("Time taken to generate random numbers through linear congruential generator is %e"%(t1))
print("Time taken to generate random numbers through np.random.rand() is %e"%(t2))
print("Note: Time is averaged over %d runs"%(N))
