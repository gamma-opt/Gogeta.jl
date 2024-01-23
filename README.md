# Gogeta

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gamma-opt.github.io/Gogeta.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gamma-opt.github.io/Gogeta.jl/dev/)

This package enables representing machine-learning models with mathematical programming, more specifically as mixed-integer optimization problems.

Currently supported models include neural networks using ReLU activation and tree ensemble models (random forests and gradient-boosted trees).

## Package features

### Decision trees
* **tree ensemble to MIP conversion** - obtain an integer optimization problem from a trained tree ensemble model
* **tree ensemble optimization** - optimize a trained decision tree model, i.e., find an input that maximizes the forest output

#### Workflow
trained tree ensemble model $\rightarrow$ TEModel $\rightarrow$ JuMP model $\rightarrow$ optimization

### Neural networks
* **neural network to MILP conversion** - convert neural networks to integer programming problems
* **bound tightening** - improve computational feasibility by tightening bounds in the formulation according to input bounds 
    * single-threaded, multithreaded and distributed
* **neural network compression** - reduce network size by removing unnecessary nodes
* **neural network optimization** - find the input that maximizes the neural network output

#### Workflow
trained Flux NN model $\rightarrow$ JuMP model $\rightarrow$ bound tightening $\rightarrow$ compression $\rightarrow$ optimization