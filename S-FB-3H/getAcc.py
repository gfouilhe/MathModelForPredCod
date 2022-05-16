import numpy as np
import os
path = os.path.join('accuracies','accTrainingCE.npy')
acc = np.load(path)

print(acc[:,-1,-1])

