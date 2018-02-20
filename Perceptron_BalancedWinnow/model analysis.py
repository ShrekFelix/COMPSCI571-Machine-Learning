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
