# library
import numpy as np

# import data from MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

# filter for 4 and 9
S=[]
for i in range( len(mnist.train.label) ):
    if mnist.train.label[i] == 4:
        S.append( (mnist.train.image[i], -1) ) #classify 4 to -1
    elif mnist.train.label[i] == 9:
        S.append( (mnist.train.image[i], 1) ) #classify 9 to 1

def perceptron(S, I):
    w = [0 for d in range( len(S[0][0]) )]
    for e in range(I):
        for i in range(len(S)):
            if dot(S[i][1], dot(w, S[i][0])) <= 0:
                w += dot(S[i][1], S[i][0])
    print(w)

perceptron(S, 100)
