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


def is_not_zero(x,tol=10**-3):
    if (abs(x) < tol).all():
        print('yo')
        return 1
    else:
        return 0

a = 0.01
d = 196
b = 0.33
l = 0.33

step = 0.05
nuimrange = np.arange(-20, 20, step)
nurerange = np.arange(-20, 20, step)
x, y = np.meshgrid(nurerange, nuimrange)
nu = complexHD(x,y)
theta = np.log((l*np.exp(nu) + (1 - b - l) + a/d* np.exp(-nu) - a/d)/(1-b*np.exp(-nu)))
equation = abs(np.real(theta)) + abs(is_not_zero(nu))
plt.contour(x, y, equation, [0])
plt.xlabel('$\Re ( \\nu )$')
plt.ylabel('$\Im ( \\nu )$')
plt.title('$\Re (\\theta) = 0$')
plt.text(7,8,f' Beta = {b} \n Lambda = {l} \n Alpha = {a} \n d = {d}')
plt.show()