# Mathematical Models For Predictive Coding

This repository contains code for studying diverse mathematical properties of predictive coding deep neural networks.

xH -> x Hidden Layers

FB -> Feedback + feedforward (full PC model)

FF -> Feedforward

S -> Simple two-circles dataset


## Introduction

Predictive Coding is a popular framework in neurosciences for explaining cortical function. In this model, higher-level cortical areas try to predict lower-level neural activity using feedback connections and prediction errors are passed back to higher layers using feedforward connections. This theory is widely supported by observations and the feedback dynamic described is likely to play a keyrole in robustness. This motivates the attempt to bring this theory in the field of Machine Learning, where Deep Neural Networks generally only features feedforward connections.

With other students, i provided a general overview of Predictive Coding with a computer science perspective in [Predictive Coding for Deep Neural Networks](LINK1).

We will here mainly study mathematically the reccurence equation of Predictive Coding : 

<img src="https://render.githubusercontent.com/render/math?math={\color{white}\e^{n+1}_j = \beta Wf_{j-1}e^{n+1}_{j-1} + \lambda Wb_{j+1}e^{n}_{j+1} + (1 - \beta - \lambda ) e^{n}_{j} - \alpha \nabla E^{n}_{j-1}">

## References :
- Predictive Coding for Deep Neural Networks, G. Fouilhé et al. *TIR Work, not published* 
  - LINK1
  - LINK2
- Predify: Augmenting deep neural networks with brain-inspired predictive coding dynamics, B. Choksi et al. https://arxiv.org/abs/2106.02749
  - code : https://github.com/bhavinc/predify2021 and https://github.com/miladmozafari/predify
- On the role of feedback in visual processing: a predictive coding perspective,  A. Alamia et al. https://arxiv.org/abs/2106.04225
  - code : https://github.com/artipago/Role_of_Feedback_in_Predictive_Coding  
- On neural differential equations, P. Kidger https://arxiv.org/abs/2202.02435
- Alpha oscillations and traveling waves: Signatures of predictive coding ?, A. Alamia & R. VanRullen https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000487
