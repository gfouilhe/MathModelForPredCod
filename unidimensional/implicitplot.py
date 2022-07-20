from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import os

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
d = 1
b = 0.33
l = 0.33

# # step = 0.05
# # nuimrange = np.arange(-20, 20, step)
# # nurerange = np.arange(-20, 20, step)
# # x, y = np.meshgrid(nurerange, nuimrange)
# nu = complexHD(x,y)


betaR = list(np.arange(0,1,0.1))
lambdaR = list(np.arange(0,1,0.1))

# blR = [(0.1,0.8), (0.1,0.5), (0.1,0.1),(0.33,0.33),(0.2,0.5),(0.5,0.2),(0.5,0.1),(0.8,0.1)]
alphaR = list(np.arange(0,1,0.1))

k = 1
thetaR = np.linspace(-np.pi,np.pi,100)
print(thetaR[0], thetaR[-1])
print(np.exp(-np.complex(0,thetaR[0])),np.exp(-np.complex(0,thetaR[-1])))

img = 0

count = 0
plt.figure()
for b in betaR:
    for l in lambdaR:
# for b, l in blR:
        for a in alphaR:
            if b+l > 1:
                pass
            else:
                img+=1
                
                # ax = plt.gca()
                # fig = plt.gcf()
                # ax.cla()
                # circle = plt.Circle((0,0),1,color='r',fill=False)
                # ax.add_patch(circle)
                for theta in thetaR:
                    nu = np.complex(0,theta)
                    rho = (l*np.exp(nu) + (1 - b - l) + a/d* np.exp(-nu) - a/d)/(1-b*np.exp(-nu))
                    x, y = np.real(rho), np.imag(rho)
                    # plt.scatter(x,y,s=0.1,c='b')
                    if x**2 + y**2 > 1:
                        count += 1
                        print('theta = ', theta)
                        print(x**2 + y**2, rho)
                        plt.scatter(b,l,c='b', s=0.5)
                        # print(x**2 + y**2)
                        # sleep(1)
                
                # plt.xlabel('$\Re ( \\rho )$')
                # plt.ylabel('$\Im ( \\rho )$')
                # plt.title('$\Im (\\theta) \\neq 0$ & $\Re (\\theta) = 0$ ')
                # plt.text(-0.7,0,f' Beta = {b:.2f} \n Lambda = {l:.2f} \n Alpha = {a:.2f} \n d = {d:.2f}')
                # plt.savefig(os.path.join('curves',f'{img}-{b:.2f}-{l:.2f}.png'))
                # plt.xlim(0.99,1.01)
                # plt.ylim(-0.01,0.01)
                # plt.text(1.005,0.005,f' Beta = {b} \n Lambda = {l} \n Alpha = {a} \n d = {d}')
                # plt.savefig(os.path.join('curves',f'{img}-{b}-{l}-zoomed.png'))

print(count)
plt.xlabel('$\\beta$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.ylabel('$\lambda$')
plt.title('$\\rho$ can go out of unit circle close to $\\theta = 0$')
plt.show()


# plt.figure()
# for b in betaR:
#     for l in lambdaR:
#         for a in alphaR:

#             if b+l > 1:
#                 pass
#             else:
#                 # gamma = (l-a/d)/(1-b)
#                 # khi = b/(1-b)
#                 # b_1 = -(gamma**2 - gamma + khi - khi**2)
#                 b_1 = - (1/2*(d**2*l**2 + a**2 + (d**2*l + a*d)*b - a*d - (2*a*d + d**2)*l)/(b**2*d**2 - 2*b*d**2 + d**2) - 1/2*b/(b**2 - 2*b + 1))
#                 a_1 = (l - b - a/d)/(b-1)
#                 if a_1 < 0:
#                     plt.scatter(b,l,c='r',s=0.5)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.xlabel('$\\beta$')
# plt.ylabel('$\lambda$')
# plt.title('$\\alpha_1 < 0$')
# plt.show()

# 1/2*(d**2*l**2 + a**2 + (d**2*l + a*d)*b - a*d - (2*a*d + d**2)*l)/(b**2*d**2 - 2*b*d**2 + d**2) - 1/2*b/(b**2 - 2*b + 1)