# Mathematical Models For Predictive Coding

This repository contains code for studying diverse mathematical properties of predictive coding deep neural networks.

xH -> x Hidden Layers

FB -> Feedback + feedforward (full PC model)

FF -> Feedforward

S -> Simple two-circles dataset

MNIST-MxM -> MNIST dataset with image size (MxM)


## Introduction

Predictive Coding is a popular framework in neurosciences for explaining cortical function. In this model, higher-level cortical areas try to predict lower-level neural activity using feedback connections and prediction errors are passed back to higher layers using feedforward connections. This theory is widely supported by observations and the feedback dynamic described is likely to play a keyrole in robustness. This motivates the attempt to bring this theory in the field of Machine Learning, where Deep Neural Networks generally only features feedforward connections.

With other students, i provided a general overview of Predictive Coding with a computer science perspective in [Predictive Coding for Deep Neural Networks](https://raw.githubusercontent.com/gfouilhe/MathModelForPredCod/main/TIR/Rapport.pdf).

We will here mainly study mathematically the reccurence equation of Predictive Coding : 

    e^{n+1}_j = \beta Wf_{j-1}e^{n+1}_{j-1} + \lambda Wb_{j+1}e^{n}_{j+1} + (1 - \beta - \lambda ) e^{n}_{j} - \alpha \nabla E^{n}_{j-1}

for all j,n.

where *n* are timesteps, *j* layers, *e* activations, *E* error of prediction, *Wf* and *Wb* weights.

## Asymptotic behaviour

We can rewrite previous reccurence equation vectorially, representing the activity of all layers on a single equation:

        e^{n+1} = \beta Wf e^{n+1} + \lambda Wb e^{n} + D e^{n} - \alpha / d E e^{n} + S
        
with appropriates Wf, Wb, D, E, D. 

This is a arithmetico-geometric reccurence equation whose asymtotoc behaviour is determined by the spectral radius of the matrix :
    
        A := (I -  \beta Wf)^-1 (\lambda Wb + D + \alpha / d E)
        
One of the main goals of the code of this repository is to study the actual properties of this matrix on a real machine learning model.

## Time and space continuous model

The reccurence equation suggest a dynamic depending simultaneously on time and space. Here space represents layers. 
Space-continuous DNNs has been recently widely studied under the name of "Neural Differential Equations". 

## References :
- Predictive Coding for Deep Neural Networks, G. Fouilhé et al. *TIR Work, not published* 
  - [report](https://raw.githubusercontent.com/gfouilhe/MathModelForPredCod/main/TIR/Rapport.pdf)
  - [poster](https://raw.githubusercontent.com/gfouilhe/MathModelForPredCod/main/TIR/Poster.pdf)
- Predify: Augmenting deep neural networks with brain-inspired predictive coding dynamics, B. Choksi et al. https://arxiv.org/abs/2106.02749
  - code : https://github.com/bhavinc/predify2021 and https://github.com/miladmozafari/predify
- On the role of feedback in visual processing: a predictive coding perspective,  A. Alamia et al. https://arxiv.org/abs/2106.04225
  - code : https://github.com/artipago/Role_of_Feedback_in_Predictive_Coding  
- On neural differential equations, P. Kidger https://arxiv.org/abs/2202.02435
- Alpha oscillations and traveling waves: Signatures of predictive coding ?, A. Alamia & R. VanRullen https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000487
