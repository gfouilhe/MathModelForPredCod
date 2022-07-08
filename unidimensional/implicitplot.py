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
l = 0.33

# # step = 0.05
# # nuimrange = np.arange(-20, 20, step)
# # nurerange = np.arange(-20, 20, step)
# # x, y = np.meshgrid(nurerange, nuimrange)
# nu = complexHD(x,y)


betaR = [0.2] #list(np.arange(0,1,0.1))
lambdaR = [0.2] #list(np.arange(0,1,0.1))
alphaR = [0.01] #,0.1,1]

thetaR = np.linspace(0,2*np.pi,50)


for b in betaR:
    for l in lambdaR:
        for a in alphaR:

            ax = plt.gca()
            fig = plt.gcf()
            ax.cla()
            circle = plt.Circle((0,0),1,color='r',fill=False)
            ax.add_patch(circle)
            for theta in thetaR:
                nu = np.complex(0,theta)
                rho = (l*np.exp(nu) + (1 - b - l) + a/d* np.exp(-nu) - a/d)/(1-b*np.exp(-nu))
                x, y = np.real(rho), np.imag(rho)
                plt.scatter(x,y)
            
            plt.xlabel('$\Re ( \\rho )$')
            plt.ylabel('$\Im ( \\rho )$')
            #plt.title('$\Im (\\theta) \\neq 0$ & $\Re (\\theta) = 0$ ')
            plt.text(0.5,0.5,f' Beta = {b} \n Lambda = {l} \n Alpha = {a} \n d = {d}')
            plt.show()