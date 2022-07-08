import matplotlib.pyplot as plt
import numpy as np

def complexHD(x,y):
    M,N = x.shape
    A,B = y.shape
    assert (A,B) == (M,N)
    new = np.zeros((M,N),dtype=complex)
    for m in range(M):
        for n in range(N):
            new[m,n] = np.complex(x[m,n],y[m,n])
    return new


def is_not_zero(x,tol=10**-6):
    M,N = x.shape
    new = np.zeros((M,N))
    for m in range(M):
        for n in range(N):
            if (abs(x[m,n]) < tol):
                new[m,n] = 1
    return new

a = 0.01
d = 196
b = 0.33
l = 0

step = 0.05
nuimrange = np.arange(-20, 20, step)
nurerange = np.arange(-20, 20, step)
x, y = np.meshgrid(nurerange, nuimrange)
nu = complexHD(x,y)


betaR = [0] #list(np.arange(0,1,0.1))
lambdaR = list(np.arange(0,1,0.1))
alphaR = [0.01,0.1,1]

for b in betaR:
    for l in lambdaR:
        for a in alphaR:

            
            theta = np.log((l*np.exp(nu) + (1 - b - l) + a/d* np.exp(-nu) - a/d)/(1-b*np.exp(-nu)))
            equation = abs(is_not_zero(np.imag(theta))) + abs(np.real(theta))
            plt.figure()
            plt.contour(x, y, equation, [0])
            plt.xlabel('$\Re ( \\nu )$')
            plt.ylabel('$\Im ( \\nu )$')
            plt.title('$\Im (\\theta) \\neq 0$ & $\Re (\\theta) = 0$ ')
            plt.text(7,8,f' Beta = {b} \n Lambda = {l} \n Alpha = {a} \n d = {d}')
            plt.show()