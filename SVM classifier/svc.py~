import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import numpy as np

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

def radial_basis_kernel(x,z,sigma_sqr):
    return np.exp(- np.linalg.norm(x-z) / sigma_sqr)

class SVC():
    def __init__(self, kernel=None, C=0):
        self.C = C
        if kernel == None:
            self.kernel = lambda x,z : np.dot(x,z)
        else:
            self.kernel = kernel
    
    def train(self, X, Y):
        n = len(Y)
        P = matrix([[(self.kernel(X[i],X[j]) * Y[i]*Y[j]) for i in range(n)] for j in range(n)])
        q = matrix([-1. for i in range(n)])
        G = matrix(-np.eye(n))
        h = matrix(np.zeros((n,1)))
        A = matrix(Y).trans()
        b = matrix(0.)
        sol = solvers.qp(P, q, G, h, A, b)
        
        self.w = (np.array(sol['x']) * Y.reshape(-1, 1)).T @ X
        self.b = - ( np.min(X[Y==1] @ w.T) + np.max(X[Y==-1] @ w.T) )/2
        
    def predict(self, X, Y=None):
        '''
        make prediction. If label is given, analyse the model.
        '''
        scores = X @ self.w.T + self.b
        Y_hat = [1 if score>0 else -1 for score in scores]
        
        
        if Y != None: #label is given, do analysis
            #confusion matrix
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in range(len(X)):
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
            
            accuracy = (TP + TN)/len(X)
            CM = {
                'TP':TP,
                'FP':FP,
                'FN':FN,
                'TN':TN
                }
            
            #ROC
            curve = sorted(zip(scores,Y), key=lambda x:x[0], reverse=True)
            x = [0 for i in range(len(curve))]
            y = [0 for i in range(len(curve))]
            for i in range(1,len(curve)):
                if curve[i][1]==1:
                    x[i] = x[i-1]
                    y[i] = y[i-1]+1/(TP+FN)
                else:
                    x[i] = x[i-1]+1/(TN+FP)
                    y[i] = y[i-1]  
            plt.plot(x,y)
            plt.xlim(0,1)
            plt.ylim(0,1)
            
            #AUC
            area = 0
            for i in range(1,len(x)):
                area += (x[i] - x[i-1]) * y[i]
            area
            #output
            prediction = {
                        'X':X,
                        'scores':scores,
                        'prediction':Y_hat,
                        'accuracy':accuracy,
                        'Confusion Matrix':CM,
                        'Area Under Curve':area
                         }
            return prediction
        
        return Y_hat
