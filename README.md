# Gogeta.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gamma-opt.github.io/Gogeta.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gamma-opt.github.io/Gogeta.jl/dev/)
![Runtests](https://github.com/gamma-opt/Gogeta.jl/workflows/CI/badge.svg)

*"Gogeta was the result of the Saiyan warriors Son Goku and Vegeta successfully performing the Fusion Dance. Vegeta and Goku usually fused into Gogeta to counteract a significant threat, as Gogeta's power exponentially surpassed the sum of his parts."* [source](https://hero.fandom.com/wiki/Gogeta)

Gogeta.jl (pronounced "Go-gee-ta") enables the user to represent machine-learning models with mathematical programming, more specifically as mixed-integer optimization problems. This, in turn, allows for "fusing" the capabilities of mathematical optimisation solvers and machine learning models to solve problems that neither could solve on their own.

Currently supported models include neural networks using ReLU activation and tree ensembles.

## Package features

### Tree ensembles
* **tree ensemble to MIP conversion** - obtain an integer optimization problem from a trained tree ensemble model
* **tree ensemble optimization** - optimize a trained decision tree model, i.e., find an input that maximizes the ensemble output

### Neural networks
* **neural network to MIP conversion** - formulate integer programming problem from a neural network
* **bound tightening** - improve computational feasibility by tightening bounds in the formulation according to input/output bounds
* **neural network compression** - reduce network size by removing inactive or stably active neurons
* **neural network optimization** - find the input that maximizes the neural network output
