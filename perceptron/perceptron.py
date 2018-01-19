# library
import numpy as np
import matplotlib.pyplot as plt

def perceptron(S, I):
    '''
takes as an input a training set S
 S: [(x, y)(x,y)...(x,y)]
      |  |
      |  S[i][1]:labels
      S[i][0]:images
after a maximum number of epochs I is reached,
return w, accuracy
       |     |
       |  list of accuracy for each epoch
    weight vector
'''
    accuracy = []
    w = [0 for d in range( len(S[0][0]) )]
    for e in range(I):
        mistakes = 0
        for i in range(len(S)):
            if S[i][1] * np.dot(w, S[i][0]) <= 0:
                mistakes += 1
                w += np.dot(S[i][1], S[i][0])
        accuracy.append( 1 - mistakes / len(S) )
        if mistakes == 0: # converges
            break
    return w, accuracy
    
def confusion_matrix(data, w, b=0):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(data)):
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

# homework problems
#
# (a). Run the function perceptron on the training set and plot the evolution of the accuracy versus the epoch counter.
w_train, acc_train = perceptron(train, 1000)
plt.figure()
plt.plot(range(len(acc_train)), acc_train)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

# (b). Plot the evolution of testing dataset accuracy versus the epoch counter (use the same figure as in part (a)).
w_test, acc_test = perceptron(test, 100)
plt.figure()
l_train, = plt.plot(range(len(acc_train)), acc_train)
l_test, = plt.plot(range(len(acc_test)), acc_test)
plt.legend(handles=[l_train, l_test], labels=['train', 'test'], loc='best')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

# (c). Report the accuracy and confusion matrix of the perceptron algorithm on the testing set after the last epoch.
print('accuracy: ',acc_test[-1])
print('confusion matrix: ', confusion_matrix(test, w_test))

# (d).


w_prime = perceptron(test[:round(len(test)/3)], 1)[0]
x_prime, y_prime = ROC(test[:round(len(test)/3)], w_prime, -1000, 1000, 1000)
w_star = perceptron(test, 100)[0]
x_star, y_star = ROC(test, w_star, -1000, 1000, 1000)
plt.figure()
l_prime, = plt.plot(x_prime, y_prime)
l_star, = plt.plot(x_star, y_star)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(handles=[l_prime, l_star], labels=['w_prime', 'w_star'], loc='best')
plt.show()


# (e).
print( AUC(x_prime, y_prime) )
print( AUC(x_star, y_star) )
