# Gogeta.jl

[Gogeta](https://gamma-opt.github.io/Gogeta.jl/) is a package that enables the user to formulate trained machine learning (ML) models as mathematical optimization problems. This approach can be utilized in the global optimization of the ML models, or when using the ML models as surrogates in larger optimization problems.

Currently supported models are $ReLU$-activated neural networks (dense and convolutional), input convex neural networks (ICNNs), and tree ensemble models.

## Installation

The latest official version can be installed from the Julia General repository.

```julia-repl
julia> Pkg.add("Gogeta")
```

Some experimental features may have been implemented on the GitHub page. The latest development version can be accessed by adding the GitHub HTTPS and the branch name as follows:

```julia-repl
julia> Pkg.add("https://github.com/gamma-opt/Gogeta.jl.git#<branch-name>")
```

Replace `<branch-name>` with the name of the branch you want to add.

!!! warning

    The code on some of the Git branches might be experimental and not work as expected.

## How can this package be used?

Formulating trained machine learning (ML) models as mixed-integer linear programming (MILP) problems opens up multiple possibilities. Firstly, it allows for global optimization - finding the input that probably maximizes or minimizes the ML model output. Secondly, changing the objective function in the MILP formulation and/or adding additional constraints makes it possible to solve problems related to the ML model, such as finding adversarial inputs. Lastly, the MILP formulation of a ML model can be embedded into a larger optimization problem. This is useful in a surrogate modeling context where an ML model is trained to approximate a complex function that itself cannot be used in an optimization problem.

Despite its usefulness, modeling ML models as MILP problems has significant limitations. The biggest limitation is the capability of MILP solvers which limits the ML model size.  With neural networks, for example, only models with at most hundreds of neurons can be effectively formulated as MILPs and then optimized. In practice, formulating into MILPs and optimizing all large modern ML models such as convolutional neural networks and transformer networks is computationally infeasible. However, if small neural networks are all that is required for the specific application, the methods implemented in this package can be useful. Secondly, only piecewise linear ML models can be formulated as MILP problems. For example, with neural networks this entails using activation functions such as $ReLU$.

Input convex neural networks (ICNNs) are a special type of machine learning model that can be formulated as linear optimization problems (LP). The convexity limits the expressiveness of the ICNN but the LP formulation enables fast optimization of even very large ICNNs. If the data or the function being modeled is approximately convex, ICNNs can provide similar accuracy to regular neural networks. If a ML model is used in some of the contexts mentioned in the first paragraph, ICNNs can be used instead of neural networks without the computational limitations of MILP models.

## Getting started

The following Features-section gives simple demonstrations on how to use the package. 
More examples and detailed code can be found in the `examples/`-folder of the [package repository](https://github.com/gamma-opt/Gogeta.jl).