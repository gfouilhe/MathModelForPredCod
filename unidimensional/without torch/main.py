from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('seaborn')
#params
b = 0.5
l = 0.2
a = 0.9
d = 1

timesteps = 500
J = 500

#init
e = np.zeros((timesteps,J+1))

#Dirichlet 
e_0 = 0
e[:,0] = np.repeat(e_0,timesteps)


e[:,J//2] = 1

#Neumann
e[:,J] = e[:,J-1]


#PC algorithm
for n in range(timesteps-1):
    for j in range(1,J):
        e[n+1,j] = b*e[n+1,j-1] + l*e[n,j+1] +(1-b-l)* e[n,j] + a/d * (e[n,j-1] - e[n,j])
    #Neumann
    e[n+1,J] = e[n+1,J-1] 



plt.figure()

for n in range(timesteps):
    if n%5 == 0:
        plt.plot(e[n,:],color = plt.get_cmap('magma')(float(n)/timesteps))
plt.ylabel('$\\epsilon$')
plt.ylim(0,0.1)
plt.xlim(0,J)
plt.xlabel(f'Layers (0 < j < {J})')
plt.title(f'Solution over {timesteps} time iterations')
plt.text(3/4*J,0.07,f" $\\beta = {b} $\n $\lambda = {l} $\n $\\alpha = {a} $\n $d = {d} $")
plt.show()


ax = plt.figure().add_subplot(projection="3d")

for n in range(timesteps):
    if n%10 == 0:
        ax.plot(np.array(range(J+1)),e[n,:],zs = n, zdir ='y',color = plt.get_cmap('magma')(float(n)/timesteps))

ax.set_xlim(0,J)
ax.set_zlim(0,0.05)
ax.set_ylim(0,timesteps)
ax.set_xlabel('j')
ax.set_ylabel('n')
ax.set_zlabel('$\epsilon$')


plt.show()