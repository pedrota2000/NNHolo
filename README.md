# NNHolo #

Paper I: [![arXiv](https://img.shields.io/badge/arXiv-2403.14763-b31b1b.svg)](https://arxiv.org/abs/2403.14763) Gravitational duals from equations of state

Paper II: [![arXiv](https://img.shields.io/badge/arXiv-2606.30117-b31b1b.svg)](https://arxiv.org/abs/2606.30117) Gravitational Duals from Equations of State II: Large Hierarchies and False Vacua

## Introduction

NNHolo is a Deep-Learning tool based on physics-informed neural networks (PINNs) built with the [neurodiffeq](https://github.com/NeuroDiffGym/neurodiffeq) package aimed to solve inverse problems in the field of holography and the AdS/CFT correspondence. In particular, it is constructed to recover the bulk geometry taking as inputs the thermodynamical properties of a given Gauge Theory.

A detailed discussion of the method and its applications can be found in [paper I](https://arxiv.org/abs/2403.14763) (thermodinamical equations of state showing crossovers and mild first and second order phase transitions) and [paper II](https://arxiv.org/abs/2606.30117) (solving for more agressive phase transitions, including large hierarchies and false vacua).


## How does it work?

NNHolo is aimed to recover a scalar potential $V(\phi)$ in the bulk for a holographic theory which is Einstein's gravity + scalar field in AdS exclusively from data obtained from the boundar theory, using PINNs.

(Direct problem): By constructing all the possible black brane solutions and solving the Einstein's equations (non-linear second order ordinary differential equations), one is able to read off the temperature $T$ and the entropy $S$ from the value of the solutions on the black brane horizon, thus obtaining the equation of state of the dual QFT at the boundary of AdS given by the relation $S(T)$. This can be done using traditional numerical solvers.

(Inverse problem): However, trying to get the scalar field potential $V(\phi)$ by only knowing the boundary conditions set by the equation of state $S(T)$ is a much more difficult problem. For this purpose, the PINN implementation in our code finds the solution to the differential equations and recovers the scalar field potential by taking as input pairs of points of the entropy as a function of the temperature. To do so, we have used solution bundles (implemented in neurodiffeq) to solve the differential equations for different boundary conditions (that are given by the points along S(T)) at the same time. Once a solution is found, another NN is introduced to make a guess for the scalar potential $V(\phi)$. Thus, all these functions are then substituted in the differential equations to compute the residuals. The square of the residuals defines then the loss function of our model. The optimization proccess to minimize the loss function is done using Adam optimizer.

- In [paper I](https://arxiv.org/abs/2403.14763), this code is used to sucessfully recover the bulk scalar potential from boundary data $S(T)$ corresponding to crossover, and first and second order phase transitions. Stronger first order phase transitions are very challenging for the code of paper I.

- In [paper II](https://arxiv.org/abs/2606.30117), an improved, more complex version of the code is used to sucessfully recover the scalar potential from boundary data $S(T)$ corresponding to strong first order phase transitions in regimes characterized by large hierarchies and the presence of false vacua.



## How to run the code
The codes are placed inside the paper_1 and paper_2 folders, respecively. 

- Inside paper_1/master, a python file called NNHolo can be found. In thise file we have include a class that, given thermodymic curve, defines the model and prepares it for the training proccess.
  
- Inside paper_2, there are two regimes: near-false-vacuum, and false vacuum.
  
    - For the near-false-vacuum regime ($\phi_M=0.8$), the code pipeline is defined in _NNholo_near_FV.py_ and called in _train.py_. The training is performed similarly to the paper I models.
      
    - For the false vacuum regime ($\phi_M = 0.55$), the training of the model is performed in two phases corresponding to the first branch (FB) and second branch (SB) of the equation of state curve $S(T)$. One finds a _FB_main_pipeline.py_ and _FB_training.ipynb_ for the FB phase. For the SB phase, the pipeline is defined in _SB_main_pipeline.py_ and called in _SB_train_model.py_. For an analysis of the results of the SB training phase, we include also a _run_and_analyze_SB_pipeline.ipynb_ notebook.


For more details, see [paper I](https://arxiv.org/abs/2403.14763) and [paper II](https://arxiv.org/abs/2606.30117), respectively.

The pretrained models are available upon request to the authors.

