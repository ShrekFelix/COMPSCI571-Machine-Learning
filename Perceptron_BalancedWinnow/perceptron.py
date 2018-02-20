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
