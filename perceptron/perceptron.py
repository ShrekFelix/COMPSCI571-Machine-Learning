# library
import numpy as np
import matplotlib.pyplot as plt

def perceptron(S, I):
    list_mistakes = []
    w = [0 for d in range( len(S[0][0]) )]
    for e in range(I):
        mistakes = 0
        for i in range(len(S)):
            if np.dot(S[i][1], np.dot(w, S[i][0])) <= 0:
                mistakes += 1
                w += np.dot(S[i][1], S[i][0])
        list_mistakes.append( mistakes / len(S) )
    plt.figure()
    plt.plot(range(I), list_mistakes)
    plt.show()

# import data from MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

# filter for 4 and 9
S=[]
for i in range( len(mnist.train.labels) ):
    if mnist.train.labels[i] == 4:
        S.append( (mnist.train.images[i], -1) ) #classify 4 to -1
    elif mnist.train.labels[i] == 9:
        S.append( (mnist.train.images[i], 1) ) #classify 9 to 1



perceptron(S, 100)
