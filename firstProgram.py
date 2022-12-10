import numpy as np

#standard normalization
#range from -3.5 to 3.5
def stdNormalization(arr):             
    z = (arr - np.mean(arr)) / np.std(arr)
    return z

#Frobenius normalization
#range from  0   to 1
def FrobNormalization(arr):                 
    z = arr/np.linalg.norm(arr)
    return z
    
def sigmoid(x):
    return 1 /(1 + np.exp(-x))

def vectorizedMultiply(arr, arr2):
    res = np.zeros((len(arr), len(arr2)))
    for i in range(len(arr)):
        res[i] = arr[i] * arr2
    return res

def MSE(modelOutput, realOuput, number):
    return np.sum((realOuput - modelOutput)**2) / number

def feedForward(wh, wo, x):
    ah = sigmoid(np.dot(wh, x))
    ao = sigmoid(np.dot(wo, ah))
    return ah, ao

def backPropagation(ah, ao, wo, wh):
    err_o = (ao - y) * ao * (1 - ao)
    err_h = np.dot(err_o, wo) * ah * (1 - ah)
    wo = wo - alpha * vectorizedMultiply(err_o, ah)
    wh = wh - alpha * vectorizedMultiply(err_h, x)
    return wo, wh

whFile = 'whFile.txt'
woFile = 'woFile.txt'

if __name__ == "__main__":
    #initializations
    inputFile = 'train.txt'
    outFile = 'result.txt'
    m,l,n,k = np.fromfile(inputFile, dtype=int, count=4, sep=" ")
    data = np.genfromtxt(inputFile, skip_header=2)
    wh = np.random.uniform(0, 1, size=(l, m))
    wo = np.random.uniform(0, 1, size=(n, l))
    epochs = 500
    alpha = 0.0003
    epoch_ao = np.zeros((k, n))
    epoch_y = data[:k, m:]
    all_mse = np.zeros(epochs)

    #standard normalization for x-columns 
    for i in range(m):
        data[:,i] = stdNormalization(data[:,i])  #range from -3.5 to 3.5
    #Frobenius normalization for y-columns, which starts from index m
    for i in range(n):
        data[:,m+i] = FrobNormalization(data[:,m+i]) 

    #Loop over epochs
    for epoch in range(epochs):
        mse = 0
        #Loop over training examples
        for i in range(k):
            x = data[i][:m]
            y = data[i][m:]
            ah, ao = feedForward(wh, wo, x)
            wo, wh = backPropagation(ah, ao, wo, wh)
            epoch_ao[i] = ao
            mse += MSE(ao, y, n)   
        all_mse[epoch] = mse / k

    np.savetxt(outFile, all_mse, fmt="%s")
    np.savetxt(whFile, wh, fmt="%s")
    np.savetxt(woFile, wo, fmt="%s")