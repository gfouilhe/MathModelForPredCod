
import torch
from model import PCMLP
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



alpha = 0.01
beta = 0.5
gamma = 0.2
mem = 0.33
model= PCMLP(0.33,alpha,beta,gamma)
checkpointPhase = torch.load(os.path.join('models',f"PC_E19_I0.pth"))
model.load_state_dict(checkpointPhase["module"])
for name, p in model.named_parameters():
    if name=='fcAB.weight':
        Wab = p.detach().numpy()
    if name=='fcBA.weight':
        Wba = p.detach().numpy()
    if name=='fcBC.weight':
        Wbc = p.detach().numpy()
    if name=='fcCB.weight':
        Wcb = p.detach().numpy()

d = 196


A11 = (1-beta-gamma) * np.eye(d)
A12 = beta * Wba
A13 = np.zeros((d,d))
A21 = (1-beta-gamma) * gamma * Wba.T + alpha/d * Wba.T
A22 = gamma * beta * Wab.dot(Wba) + (1-beta-gamma) * np.eye(d) - alpha/d * Wba.T.dot(Wba)
A23 = beta * Wcb
A31 = (1-beta-gamma) * gamma**2 * Wbc.dot(Wab) + alpha/d * gamma * Wbc.dot(Wcb.T)
A32 = beta * gamma **2 * Wab.dot(Wba) + (1-beta-gamma) * gamma * Wbc - alpha/d * gamma * Wbc.dot(Wba.T.dot(Wba)) + alpha/d * Wcb.T
A33 = beta * gamma * Wbc * Wcb + (1-gamma) * np.eye(d) - alpha/d * Wcb.T.dot(Wcb)
A = np.block([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])

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

ax.set_title('Eigenvalues of $\mathbb{A}, \\alpha_{mem}, \\beta_{Bw}, \gamma_{Fw} = 0.01, 0.5, 0.2$')
plt.show()
fig.savefig('Eigenvalues.png')
print(w)
print((1-beta-gamma))
print(1-beta)

# #%%
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

d = 196
J = 3
Idj = np.eye(d*J)
Id = np.eye(d)
Z = np.zeros(d)
Wf = np.block([[Z,Z,Z],[Wab,Z,Z],[Z,Wbc,Z]])
Wb = np.block([[Z,Wba,Z],[Z,Z,Wcb],[Z,Z,Z]])
D = np.block([[(1-beta-gamma)*Id,Z,Z],[Z,(1-beta-gamma)*Id,Z],[Z,Z,(1-gamma)*Id]])
E = np.block([[-WZ,Wba,Z],[Z,Z,Wcb],[Z,Z,Z]])