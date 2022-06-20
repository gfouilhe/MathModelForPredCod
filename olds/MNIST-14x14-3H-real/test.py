import torch

X = torch.asarray([1.,2.,3.,4.,5.,6.,7.,8.])

print(X)

Y = X[:4]
X = X[4:]

print(X,Y)

Z = torch.stack((Y,X),dim=1,)
print(Z, torch.view_as_complex(Z))