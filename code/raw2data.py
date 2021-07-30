import numpy as np
import math

data = np.loadtxt('data/testraw.txt')

X = data[:,0]
Y = np.round(data[:,1]*1)
Ymin = min(Y)
# YminArray = [Ymin] * len(Y)
# print(YminArray)

Yarray = np.array(Y)
Y = Y-Ymin
# print(Y)

Z = []
n=0

for i in Y:
    num_x=Y[n]
    # print(num_x)
    j=0
    while j < num_x :
        Z.append(X[n])
        j +=1
    #     x = X[n]
    #     print(x)
    #     Z.append(x)        
    n +=1

    
# Z.append(point)
# print(Z)
# print(X,Y)

np.savetxt('data/converted_raw.txt', Z, delimiter=",")