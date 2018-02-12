# library
import numpy as np
import matplotlib.pyplot as plt
import copy

# S: [(x, y)(x,y)...(x,y)]
#      |  |
#      |  S[i][1]:labels yi
#      S[i][0]:images xi(784,)

def perceptron(S, I, converge=1):
    # initialization
    w = [0 for d in range( len(S[0][0]) )]
    W = []
    for e in range(I):
        mistakes = 0 # number of mistakes made in each epoch
        for i in range(len(S)): # iterate over dataset
            if S[i][1] * np.dot(w, S[i][0]) <= 0: # label not agree with prediction
                mistakes += 1
                w += np.dot(S[i][1], S[i][0])
        W.append(copy.deepcopy(w)) # save w after each epoch
        if ( 1 - mistakes / len(S) ) >= converge: # converges
            break
    return W

def balanced_winnow(S, I, eta=0.1, converge=1):
    p = len(S[0][0])
    wp = [1/(2*p) for i in range(p)]
    wn = [1/(2*p) for i in range(p)]
    W = []
    for e in range(I):
        mistakes = 0
        for i in range(len(S)): # iterate over dataset
            if S[i][1] * ( np.dot(wp, S[i][0]) - np.dot(wn, S[i][0]) ) <= 0: # label not agree with prediction
                mistakes += 1
                s = 0 # to normalize w
                for j in range(p): # update each element of one weight vector
                    wp[j] *= np.exp(eta * S[i][1] * S[i][0][j])
                    wn[j] *= np.exp(-eta * S[i][1] * S[i][0][j])
                    s += wp[j] + wn[j]
                # normalization
                wp = np.dot(wp, 1/s)
                wn = np.dot(wn, 1/s)
        W.append(wp - wn) # save w after each epoch
        if ( 1 - mistakes / len(S) ) >= converge: # converges
            break
    return W

def accuracy(S, W):
    accuracy = []
    for e in range(len(W)): # for each epoch
        mistakes = 0
        for i in range(len(S)): # count mistakes
            if S[i][1] * np.dot(W[e], S[i][0]) <= 0: # label not agree with prediction
                mistakes += 1
        accuracy.append( 1 - mistakes / len(S) ) # update list of accuracy for each epoch
    return accuracy
    
def confusion_matrix(data, w, b=0):
    '''
b: bias in the classification hyperplane.
Useful for computing different sets of TPR/FPR pairs and plotting ROCs.
'''
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(data)): # iterate over dataset
        y_hat = np.dot(w, data[i][0]) - b
        if y_hat * data[i][1] > 0:
            if data[i][1] > 0:
                TP += 1
            else:
                TN += 1
        else:
            if data[i][1] > 0:
                FN += 1
            else:
                FP += 1            
    return [ [TP, FP], [FN, TN] ]

def ROC(data, w, start = -10, stop = 10, num = 100):
    x = []
    y = []
    for b in np.linspace(start, stop, num):
        [ [TP, FP], [FN, TN] ] = confusion_matrix(data, w, b)
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        x.append(FPR)
        y.append(TPR)
    return x, y

def AUC(x, y):
    area = 0
    x.append(0)
    y.append(0)
    for i in range(len(x)-1):
        area += (x[i] - x[i+1]) * (y[i+1] + y[i])
    area /= 2
    return area

# import data from MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

# Datasets    DataSet         ndarray
#
# mnist ------ train -------- images(55000, 784)
#        |____ test        |_ labels(55000,)

# arrange data into training and testing group, filter for digit '9' and '4'
train =[]
for i in range( len(mnist.train.labels) ):
    # digit "4" -> -1
    if mnist.train.labels[i] == 4:
        train.append( (mnist.train.images[i], -1) )
    # digit "9" -> 1
    elif mnist.train.labels[i] == 9:
        train.append( (mnist.train.images[i], 1) )

test =[]
for i in range( len(mnist.test.labels) ):
    # digit "4" -> -1
    if mnist.test.labels[i] == 4:
        test.append( (mnist.test.images[i], -1) )
    # digit "9" -> 1
    elif mnist.test.labels[i] == 9:
        test.append( (mnist.test.images[i], 1) )
