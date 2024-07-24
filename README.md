# Gogeta.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gamma-opt.github.io/Gogeta.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gamma-opt.github.io/Gogeta.jl/dev/)
![Runtests](https://github.com/gamma-opt/Gogeta.jl/workflows/CI/badge.svg)

*"Gogeta was the result of the Saiyan warriors Son Goku and Vegeta successfully performing the Fusion Dance. Vegeta and Goku usually fused into Gogeta to counteract a significant threat, as Gogeta's power exponentially surpassed the sum of his parts."* [source](https://hero.fandom.com/wiki/Gogeta)

Gogeta.jl (pronounced "Go-gee-ta") enables the user to represent trained machine learning models with mathematical programming, more specifically as mixed-integer optimization problems. This, in turn, allows for "fusing" the capabilities of mathematical optimization solvers and machine learning models to solve problems that neither could solve on their own.

Currently supported models are tree ensembles, input convex neural networks, and neural networks and convolutional neural networks using ReLU activation.