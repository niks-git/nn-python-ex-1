import numpy as np

def nonlin(x, Deriv=False):
    if Deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0,1,1,1]]).T

np.random.seed(1)

syn1 = 2*np.random.random((4,1)) - 1
syn0 = 2*np.random.random((3,4)) - 1

for iter in range(60000):
    l0 = X                                  #l0-4,3
    l1 = nonlin(np.dot(l0,syn0))            #l1=4,4
    l2 = nonlin(np.dot(l1,syn1))            #l2=4,1
    
    l2_error = y - l2                       #l2error=4,1
    l2_delta = l2_error*nonlin(l2,True)     #l2delta=4,1
    
    l1_error = np.dot(l2_delta,syn1.T)      #l1error=4,4
    l1_delta = l1_error*nonlin(l1,True)     #l1delta=4,4
    
    syn1 += np.dot(l1.T,l2_delta)
    syn0 += np.dot(l0.T,l1_delta)
    
print ('Output after training')
print (l2)