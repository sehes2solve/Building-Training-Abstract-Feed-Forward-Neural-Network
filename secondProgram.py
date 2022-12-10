import numpy as np
import firstProgram

#initializations
inputFile = 'test.txt'
m,l,n,k = np.fromfile(inputFile, dtype=int, count=4, sep=" ")
data = np.genfromtxt(inputFile, skip_header=2)
wh = np.genfromtxt(firstProgram.whFile)
wo = np.genfromtxt(firstProgram.woFile)
epoch_ao = np.zeros((k, n))
epoch_y = data[:k, m:]
mse = 0

#standard normalization for x-columns 
for i in range(m):
    data[:,i] = firstProgram.stdNormalization(data[:,i])  #range from -3.5 to 3.5
#Frobenius normalization for y-columns, which starts from index m
for i in range(n):
    data[:,m+i] = firstProgram.FrobNormalization(data[:,m+i]) 

#Loop over training examples
for i in range(k):
    x = data[i][:m]
    y = data[i][m:]
    ah, ao = firstProgram.feedForward(wh, wo, x)
    epoch_ao[i] = ao
    mse += firstProgram.MSE(ao, y, n)
mse = mse/k
print('mse = ', mse)