import torch
from model import PCMLP
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

alpha = 0.01

betaR = list(np.arange(0,1,0.01))[1:]
gammaR = list(np.arange(0,1,0.01))[1:]

threshold = []

for beta in betaR:
    for i, gamma in enumerate(gammaR):
        if beta + gamma > 1:
            break
        A11 = (1-beta-gamma) * np.eye(5)
        A12 = beta * Wba
        A13 = np.zeros((5,5))
        A21 = (1-beta-gamma) * gamma * Wba.T + alpha/5 * Wba.T
        A22 = gamma * beta * Wab.dot(Wba) + (1-beta-gamma) * np.eye(5) - alpha/5 * Wba.T.dot(Wba)
        A23 = beta * Wcb
        A31 = (1-beta-gamma) * gamma**2 * Wbc.dot(Wab) + alpha/5 * gamma * Wbc.dot(Wcb.T)
        A32 = beta * gamma **2 * Wab.dot(Wba) + (1-beta-gamma) * gamma * Wbc - alpha/5 * gamma * Wbc.dot(Wba.T.dot(Wba)) + alpha/5 * Wcb.T
        A33 = beta * gamma * Wbc * Wcb + (1-gamma) * np.eye(5) - alpha/5 * Wcb.T.dot(Wcb)
        A = np.block([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])
        w, v = np.linalg.eig(A)
        
