# Gogeta.jl

[Gogeta](https://gamma-opt.github.io/Gogeta.jl/) is a package that enables the user to formulate trained machine learning models as mathematical optimization problems.

Currently supported models are `Flux.Chain` ReLU-activated neural networks (dense and convolutional) and `EvoTrees` tree ensemble models.

## Installation
```julia-repl
julia> Pkg.add("Gogeta")
```

## How can this package be used?

Formulating trained machine learning (ML) models as mixed-integer programming (MIP) problems opens up multiple possibilities. Firstly, it allows for global optimization - finding the input that provably maximizes or minimizes the ML model output. Secondly, changing the objective function in the MIP formulation and/or adding additional constraints makes it possible to solve problems related to the ML model, such as finding adversarial inputs. Lastly, the MIP formulation of a ML model can be included into a larger optimization problem. This is useful in surrogate contexts where an ML model can be trained to approximate a complicated function that itself cannot be used in an optimization problem.

Despite its usefulness, modeling ML models as MIP problems has significant limitations. The biggest limitation is the capability of MIP solvers which limits the ML model size.  With neural networks, for example, only models with at most hundreds of neurons can be effectively tackled. In practice, formulating into MIPs and optimizing all large modern models such as convolutional neural networks and transformer networks is computationally infeasible. However, if small neural networks are all that is required for the specific application, the techniques implemented in this package can be useful. Secondly, only piecewise linear ML models can be formulated as MIP problems. For example, with neural networks this entails using only ReLU as the activation function.

## Getting started

The following sections [Tree ensembles](tree_ensembles.md) and [Neural networks](neural_networks.md) give a very simple demonstration on how to use the package. 
Multiprocessing examples and more detailed code can be found in the `examples/`-folder of the [package repository](https://github.com/gamma-opt/Gogeta.jl).