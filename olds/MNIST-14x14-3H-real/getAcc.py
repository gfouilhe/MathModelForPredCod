import numpy as np
import os

def getMean(acc,mode='ff'):

    assert mode in ['ff','fb']
    if mode=='ff':
        mean = np.mean(acc[-1,:])
        return mean
    
    timeSteps = acc.shape[0]
    mean = np.zeros(timeSteps)
    for t in range(timeSteps):
        mean[t] = np.mean(acc[t,-1,:])
    return mean

path = os.path.join('accuracies','ACaccTrainingCE__G0.6_B0.2_A0.01.npy')
acc = np.load(path)



print('Mean FB : ', getMean(acc,mode='fb'))

path = os.path.join('accuracies','accTrainingCET__G0.6_B0.2_A0.01.npy')
acc = np.load(path)



print('Mean FB : ', getMean(acc,mode='fb'))



# path = os.path.join('accuracies','accTrainingFFCE.npy')
# acc = np.load(path)

# print('Mean FF : ', getMean(acc))