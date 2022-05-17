from scipy.sparse.linalg import eigs

import numpy as np

id = np.diag([1,3,541,-2,-5413,620005,0.0005,100009])

vals, vecs = eigs(id, k=1)

print(vals)