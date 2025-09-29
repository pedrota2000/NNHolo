# NNHolo #

[![arXiv](https://img.shields.io/badge/arXiv-2403.14763-b31b1b.svg)](https://arxiv.org/abs/2403.14763)

## Introduction

NNHolo is a Deep-Learning tool based on neurodiffeq package (https://github.com/NeuroDiffGym/neurodiffeq) aimed to solve holography inverse problems. In particular, it is constructed to recover the bulk geometry taking as inputs the thermodynamical properties of a given Gauge Theory.

A detailed discussion of the method and its applications can be found in [this paper](https://arxiv.org/abs/2403.14763).


## How does this work?

NNHolo is aimed to recover a scalar potential in the bulk for a holographic theory which is Einstein's gravity + scalar field in AdS. By constructing all the possible black brane solutions and solving the Einstein's equations (non-linear second order ordinary differential equations), one is able to read off the temperature and the entropy from the value of the solutions on the horizon. However, trying to get the scalar field potential by only knowing the boundary conditions is a much more difficult problem. 

For this purpose, our code finds the solution to the differential equations and recovers the scalar field potential by taking as input pairs of points of the entropy as a function of the temperature. To do so, we have used solution bundles (implemented in neurodiffeq) to solve the differential equations for different boundary conditions (that are given by the points along S(T)) at the same time. Once a solution is found, another NN is introduced to make a guess for the scalar potential. Thus, all these functions are then introduced in the differential equations to compute the residuals, which square is going to be the loss function of the model. The optimization proccess to minimize the loss function is done using adam optimizer.



## How to run the code
The code is placed insied the master folder, where a python file called NNHolo can be found. In thise file we have include a class that, given thermodymic curve, defines the model and prepares it for the training proccess. 


