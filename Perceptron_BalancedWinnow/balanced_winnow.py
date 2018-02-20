# library
import numpy as np
import matplotlib.pyplot as plt
import copy

# S: [(x, y)(x,y)...(x,y)]
#      |  |
#      |  S[i][1]:labels yi
#      S[i][0]:images xi(784,)

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
