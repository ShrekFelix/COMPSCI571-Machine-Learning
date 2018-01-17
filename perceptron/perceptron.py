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
after a maximum number of epochs I is reached, prints out:
a weight vector w 
the accuracy
the confusion_matrix
evolution of accuracy vs epoch
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
        
    return w, accuracy
    
def confusion_matrix(S, w):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(S)):
        y_hat = np.dot(w, S[i][0])
        if y_hat > 0 and S[i][1] > 0:
            TP += 1
        elif y_hat > 0 and S[i][1] < 0:
            FP += 1
        elif y_hat < 0 and S[i][1] > 0:
            FN += 1
        elif y_hat < 0 and S[i][1] < 0:
            TN += 1
            
    return [ [TP, FP], [FN, TN] ]
            
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


# (a). Run the function perceptron on the training set and plot the evolution of the accuracy versus the epoch counter.
w_train, acc_train = perceptron(train, 100)
plt.figure()
plt.plot(range(len(acc_train)), acc_train)
plt.show()

# (b). Plot the evolution of testing dataset accuracy versus the epoch counter (use the same figure as in part (a)).
w_test, acc_test = perceptron(test, 100)
plt.figure()
plt.plot(range(len(acc_train)), acc_train)
plt.plot(range(len(acc_test)), acc_test)
plt.show()

# (c). Report the accuracy and confusion matrix of the perceptron algorithm on the testing set after the last epoch.
print('accuracy: ',acc_test[-1])
print('confusion matrix: ', confusion_matrix(test, w_train))
