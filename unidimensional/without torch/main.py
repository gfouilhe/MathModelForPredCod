from os import times
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('seaborn')
#params
b = 0.33
l = 0.33
a = 0.01
d = 100

timesteps = 500
J = 100

#init
e = np.zeros((timesteps,J+1))

#Dirichlet 
e_0 = 0
e[:,0] = np.repeat(e_0,timesteps)

# #feedforward weights for first ff pass
# w = np.zeros(J+1) #np.random.normal(scale=0.5,size=J+1)
# #first forward pass
# for j in range(J):
#     e[:,j+1] = w[j]*e[:,j]

e[:,J//2] = 1

#Neumann
e[:,J-1] = e[:,J]


#PC algorithm
for n in range(timesteps-1):
    for j in range(1,J):
        e[n+1,j] = b*e[n+1,j-1] + l*e[n,j+1] +(1-b-l)* e[n,j] + a/d * (e[n,j-1] - e[n,j])
    #Neumann
    e[n+1,J-1] = e[n+1,J] 



plt.figure()

for n in range(timesteps):
    plt.plot(e[n,:],color = plt.get_cmap('magma')(float(n)/timesteps))
plt.ylabel('$\\epsilon$')
plt.ylim(0,0.1)
plt.xlim(0,J)
plt.xlabel(f'Layers (0 < j < {J})')
plt.title(f'Solution over {timesteps} time iterations')
plt.text(3/4*J,0.07,f" $\\beta = {b} $\n $\lambda = {l} $\n $\\alpha = {a} $\n $d = {d} $")
plt.show()
