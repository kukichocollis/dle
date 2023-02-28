# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:34:10 2022

@author: life21c
"""

# chap07-numpy-study
import numpy as np

X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
T = np.array([[0], [1], [1], [0]])
X.shape,T.shape

for x, y in zip(X, T):
    print(x.shape)
    x = np.reshape(x, (1, -1))
    print(x.shape)

############################

