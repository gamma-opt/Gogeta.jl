# Gogeta

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gamma-opt.github.io/Gogeta.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gamma-opt.github.io/Gogeta.jl/dev/)
![Runtests](https://github.com/gamma-opt/Gogeta.jl/workflows/CI/badge.svg)

This package enables the user to represent machine-learning models with mathematical programming, more specifically as mixed-integer optimization problems.

Currently supported models include neural networks using ReLU activation and tree ensembles.

## Package features

### Tree ensembles
* **tree ensemble to MIP conversion** - obtain an integer optimization problem from a trained tree ensemble model
* **tree ensemble optimization** - optimize a trained decision tree model, i.e., find an input that maximizes the ensemble output

### Neural networks
* **neural network to MIP conversion** - formulate integer programming problem from a neural network
* **bound tightening** - improve computational feasibility by tightening bounds in the formulation according to input/output bounds
* **neural network compression** - reduce network size by removing inactive or stabily active neurons
* **neural network optimization** - find the input that maximizes the neural network output