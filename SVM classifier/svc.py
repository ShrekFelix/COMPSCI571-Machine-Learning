import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import numpy as np


class SVC():
    '''
__________
parameters:
-----------
C:
float, optional (default=1.0)
Penalty parameter C of the error term.

kernel:
user defined kernel function, optional (default is linear)

____________
fields
------------
X: data features used to train this classifier
Y: data labels used to train this classifier
alpha 
b
wx: inner product of w and phi(x)

____________
methods:
----------------
train(X_train, y_train)
predict(X_test)
'''
    
    def __init__(self, kernel=None, C=1):
        self.C = C #use of C is currently not implemented
        if kernel == None:
            self.kernel = lambda X,Z : np.sum(X * Z)
        else:
            self.kernel = kernel
    
    def train(self, X, Y):
        self.X = X
        self.Y = Y.reshape(-1,1)

        n = len(Y)        
        P = matrix([[(self.kernel(X[i],X[j]) * Y[i]*Y[j]) for i in range(n)] for j in range(n)])
        q = matrix([-1. for i in range(n)])
        G = matrix(-np.eye(n))
        h = matrix(np.zeros((n,1)))
        A = matrix(Y).trans()
        b = matrix(0.)
        sol = solvers.qp(P, q, G, h, A, b)
        
        self.alpha = np.array(sol['x'])
        
        #support vector
        s = [0 for i in range(n)]
        for i in range(n):
            for j in range(n):
                s[i] += self.alpha[j] * Y[j] * self.kernel(X[j], X[i])
        self.wx = np.array(s) #inner product of w and phi(x), we can find support vectors using this array
        self.b = 1 - np.min(self.wx[Y==1]) #bias
        
    def predict(self, X):
        scores = [0 for i in range(len(X))]
        for i in range(len(X)):
            k = np.array([self.kernel(self.X[i], X[j]) for j in range(len(self.X))])
            scores[i] = np.sum(alpha * self.Y * k.reshape(-1,1)) + self.b
        
        Y_hat = [1 if score>0 else -1 for score in scores]
        return Y_hat





def load_CSV(file_name):
    tmp = np.loadtxt(file_name, dtype=np.str, delimiter=",")
    features = tmp[1:,:-1].astype(np.float)# load features
    labels = tmp[1:,-1].astype(np.float)# load labels
    return features, labels # return ndarray

def scale(data):
    return (data - data.mean()) / data.std()

def ConfusionMatrix(Y_hat, Y):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(Y)):
        if Y_hat[i]==Y[i]:
            if Y_hat[i]==1:
                TP += 1
            else:
                TN += 1
        else:
            if Y_hat[i]==1:
                FP += 1
            else:
                FN += 1
                
    CM = {
        'TP':TP,
        'FP':FP,
        'FN':FN,
        'TN':TN
    }
    return CM     

def ROC(X, Y, classifier):
    scores = classifier.decision_function(X)
    curve = sorted(zip(scores,Y), key=lambda x:x[0], reverse=True)
    x = [0 for i in range(len(curve))]
    y = [0 for i in range(len(curve))]
    for i in range(1,len(curve)):
        if curve[i][1]==1:
            x[i] = x[i-1]
            y[i] = y[i-1]+1/len(Y[Y==1])
        else:
            x[i] = x[i-1]+1/len(Y[Y==0])
            y[i] = y[i-1]  
    plt.plot(x,y)
    plt.xlim(0,1)
    plt.ylim(0,1)   
    return x,y

def AUC(x,y):
    area = 0
    for i in range(1,len(x)):
        area += (x[i] - x[i-1]) * y[i]
    return area

def radial_basis_kernel(x,z,sigma_sqr=25):
    return np.exp(- np.linalg.norm(x-z) / sigma_sqr)




