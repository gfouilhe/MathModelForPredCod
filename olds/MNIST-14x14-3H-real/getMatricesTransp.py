
import torch
from model import PCMLP
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



alpha = 0.01
beta = 0.33
gamma = 0.33
mem = 0.33
model= PCMLP(0.33,alpha,beta,gamma)
checkpointPhase = torch.load(os.path.join('models',f"PCT_E19_I0_G0.33_B0.33_A0.01.pth")) #PCT_E19_I0_G0.6_B0.2_A0.01
model.load_state_dict(checkpointPhase["module"])
for name, p in model.named_parameters():
    tmp = p.detach().numpy()
    if name=='fcAB.weight':
        W12 = tmp
    if name=='fcBA.weight':
        W21 = tmp
    if name=='fciA.weight':
        W01 = tmp
    if name=='fcAi.weight':
        W10 = tmp

W21 = W12.T
W10 = W01.T

d = W12.shape[1]
Z = np.zeros((d,d))
Wf = np.block([[Z,Z],[W12,Z]])
Wb = np.block([[Z,W21],[Z,Z]])
IdJ = np.eye(2*d)
Id = np.eye(d)

M1 = np.linalg.inv((IdJ-gamma*Wf))

D = np.block([[(1-beta-gamma)*Id,Z],[Z,(1-gamma)*Id]])
E = np.block([[-W10.T.dot(W10),Z],[W21.T,-W21.T.dot(W21)]])
M2 = beta*Wb + D + alpha/d * E

A = M1.dot(M2)

A11 = (1-beta-gamma) * np.eye(d)
A12 = beta * W21
A21 = (1-beta-gamma) * gamma * W12 + alpha/d * W21.T
A22 = gamma * beta * W12.dot(W21) + (1-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)

A_hand = np.block([[A11,A12],[A21,A22]])

print(np.max(A-A_hand))

w, v = np.linalg.eig(A)
ax = plt.gca()
fig = plt.gcf()
ax.cla()
circle = plt.Circle((0,0),1,color='r',fill=False)
ax.add_patch(circle)
for eig in w:
    re = np.real(eig)
    im = np.imag(eig)
    ax.plot(re,im,'o',color='blue')

plt.show() 
plt.close()
ax = plt.gca()
fig = plt.gcf()
ax.cla()
circle = plt.Circle((0,0),1,color='r',fill=False)
ax.add_patch(circle)
for eig in w:
    re = np.real(eig)
    im = np.imag(eig)
    ax.plot(re,im,'o',color='blue')
    
plt.xlim(0,2)
plt.show() 

plt.close()
ax = plt.gca()
fig = plt.gcf()
ax.cla()
circle = plt.Circle((0,0),1,color='r',fill=False)
ax.add_patch(circle)
for eig in w:
    re = np.real(eig)
    im = np.imag(eig)
    ax.plot(re,im,'o',color='blue')
    
plt.xlim(0.85,1.15)
plt.show()  

plt.close()
ax = plt.gca()
fig = plt.gcf()
ax.cla()
circle = plt.Circle((0,0),1,color='r',fill=False)
ax.add_patch(circle)
for eig in w:
    re = np.real(eig)
    im = np.imag(eig)
    ax.plot(re,im,'o',color='blue')
    
plt.xlim(0.98,1.02)
plt.show()
# alpha = 0.01
# beta = 0.33
# gamma = 0.33
# model= PCMLP(5,5,5,0,alpha,beta,gamma)
# checkpointPhase = torch.load(os.path.join('models',f"PC_E19_I4.pth"))
# model.load_state_dict(checkpointPhase["module"])
# for name, p in model.named_parameters():
#     if name=='fcAB.weight':
#         Wab = p.detach().numpy()
#     if name=='fcBA.weight':
#         Wba = p.detach().numpy()
#     if name=='fcBC.weight':
#         Wbc = p.detach().numpy()
#     if name=='fcCB.weight':
#         Wcb = p.detach().numpy()


# A11 = (1-beta-gamma) * np.eye(5)
# A12 = beta * Wba
# A13 = np.zeros((5,5))
# A21 = (1-beta-gamma) * gamma * Wba.T + alpha/5 * Wba.T
# A22 = gamma * beta * Wab.dot(Wba) + (1-beta-gamma) * np.eye(5) - alpha/5 * Wba.T.dot(Wba)
# A23 = beta * Wcb
# A31 = (1-beta-gamma) * gamma**2 * Wbc.dot(Wab) + alpha/5 * gamma * Wbc.dot(Wcb.T)
# A32 = beta * gamma **2 * Wab.dot(Wba) + (1-beta-gamma) * gamma * Wbc - alpha/5 * gamma * Wbc.dot(Wba.T.dot(Wba)) + alpha/5 * Wcb.T
# A33 = beta * gamma * Wbc * Wcb + (1-gamma) * np.eye(5) - alpha/5 * Wcb.T.dot(Wcb)
# A = np.block([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])


# w, v = np.linalg.eig(A)

# print(abs(w))

# # all eigenvalues modules are < 1 !!!



# betaR = list(np.arange(0,1,0.01))[1:]
# gammaR = list(np.arange(0,1,0.01))[1:]
# alphaR = list(np.arange(0,1,0.01))[1:]

# for beta in betaR:
#     for gamma in gammaR:
#         for alpha in alphaR:
#             if alpha + beta + gamma > 1:
                    
            
#             A11 = (1-beta-gamma) * np.eye(5)
#             A12 = beta * Wba
#             A13 = np.zeros((5,5))
#             A21 = (1-beta-gamma) * gamma * Wba.T + alpha/5 * Wba.T
#             A22 = gamma * beta * Wab.dot(Wba) + (1-beta-gamma) * np.eye(5) - alpha/5 * Wba.T.dot(Wba)
#             A23 = beta * Wcb
#             A31 = (1-beta-gamma) * gamma**2 * Wbc.dot(Wab) + alpha/5 * gamma * Wbc.dot(Wcb.T)
#             A32 = beta * gamma **2 * Wab.dot(Wba) + (1-beta-gamma) * gamma * Wbc - alpha/5 * gamma * Wbc.dot(Wba.T.dot(Wba)) + alpha/5 * Wcb.T
#             A33 = beta * gamma * Wbc * Wcb + (1-gamma) * np.eye(5) - alpha/5 * Wcb.T.dot(Wcb)
#             A = np.block([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])


#             w, v = np.linalg.eig(A)
#             if max(abs(w))==1:
#                 print('=1',beta,gamma,alpha, alpha+beta+gamma)
#             if max(abs(w))>1:
#                 print('>1',beta,gamma,alpha, alpha+beta+gamma)

# For this learned weights, any parameters choice beta, gamma, alpha in (0,1] s.t. sum() <=1 give the same result.


