import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.stats import chi2
import matplotlib.pyplot as plt

def chi2Test(x,dof):
  p=1-chi2.cdf(x,dof)
  if((p<0.01 )|(p>0.99)):
    print("Not sufficiently random")
  if((0.01<p<0.05)|(0.95<p<0.99)):
    print("Suspect")
  if((0.05<p<0.10)|(0.9<p<0.95)):
    print("Almost suspect")
  if(0.1<p<0.9):
    print("Sufficiently random")


count1=np.array([4, 10, 10, 13, 20, 18, 18, 11, 13, 14, 13])
count2=np.array([3, 7, 11, 15, 19, 24, 21, 17, 13, 9, 5])
expected=np.array([4, 8, 12, 16, 20, 24, 20, 16, 12, 8, 4])

n=count1.size
cchi1=cchi2=0
for i in range(n):
  cchi1+=((count1[i]-expected[i])**2)/expected[i]
  cchi2+=((count2[i]-expected[i])**2)/expected[i]
  
p1=1-chi2.cdf(cchi1,10.0)
p2=1-chi2.cdf(cchi2,10.0)


print("1)First run")
chi2Test(cchi1,10.0)
print("2)Second run")
chi2Test(cchi2,10.0)

