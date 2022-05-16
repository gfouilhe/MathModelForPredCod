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

path = os.path.join('accuracies','accTrainingCE.npy')
acc = np.load(path)



print('Mean FB : ', getMean(acc,mode='fb'))



# path = os.path.join('accuracies','accTrainingFFCE.npy')
# acc = np.load(path)

# print('Mean FF : ', getMean(acc))


# path = os.path.join('accuracies','accLinear.npy')
# acc = np.load(path)
# print('Mean Linear : ', getMean(acc,mode='fb'))
