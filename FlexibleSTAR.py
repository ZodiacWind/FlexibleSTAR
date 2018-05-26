# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:20:55 2018

@author: Tongshuai
"""

import numpy as np
import matplotlib.pyplot as plt

# N locations
N = 1000

# number of Groups
G = 4

# T different times
T = 10000

# Length of each group
groupLen = int(N / G)

phiPlusPsi = np.random.rand(G)

phi = phiPlusPsi * np.random.rand(G)
psi = phiPlusPsi - phi

# Define params of each group
phiMatrix = np.zeros(N)
psiMatrix = np.zeros(N)
for i in range(G):
    phiMatrix[i*groupLen:(i+1)*groupLen-1] = phi[i]
    psiMatrix[i*groupLen:(i+1)*groupLen-1] = psi[i]


# Define W
#W = np.random.rand(N,N)
#W = W - np.diagonal(W)
#W = W/ W.sum(0)
#W = W.T
W = 0.5 * (np.eye(N, k=1) + np.eye(N,k=-1))


# uncorrelated white noise disturbances
eta = np.random.normal(0,1,size=(N,T))

# Rho is the saclar factor
Rho = 1

# Epsilon is the input
Epsilon = Rho * np.dot(W, eta)

y = np.zeros((N, T))
y[:,0] = Epsilon[:,0]
for i in range(1, T):
    y[:,i] = phiMatrix * y[:,i-1] + psiMatrix * (np.dot(W, y[:,i-1])) + Epsilon[:,i]

plt.plot(y[:,100:200])