import numpy as np 
import os


for f in os.listdir(os.path.join('accuracies')):
    print(np.load(os.path.join('accuracies',f))[:,-1,-1])

    