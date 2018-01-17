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

    # weight vector
    print('w: ', w)
    
    # confusion matrix
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
    confusion_matrix = [ [TP, FP], [FN, TN] ]
    print("accuracy: ", accuracy[-1])
    print(confusion_matrix)
    
    # plot evolution of accuracy vs epoch
    plt.figure()
    plt.plot(range(I), accuracy)
    plt.show()
            
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



perceptron(train, 100)
perceptron(test, 100)
