import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def kernel(point,xmat, k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye((m)))  
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
    return weights
 
def localWeight(point,xmat,ymat,k):
    wei = kernel(point,xmat,k)
    W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))
    return W
     
def localWeightRegression(xmat,ymat,k):
    m,n = np.shape(xmat)    
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
    return ypred

data = pd.read_csv('tips.csv')
bill = data.total_bill.values
tip = data.tip.values

mbill = np.mat(bill)
mtip = np.mat(tip)
m= np.shape(mbill)[1] # columns

one = np.mat(np.ones(m))

X= np.hstack((one.T,mbill.T)) # horizontal stack
#set k here
ypred = localWeightRegression(X,mtip,10)

SortIndex = X[:,1].argsort(0)
xsort = X[SortIndex][:,0]
 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(bill,tip, color='green')
ax.plot(xsort[:,1],ypred[SortIndex], color = 'red', linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show()