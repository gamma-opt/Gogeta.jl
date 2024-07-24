# Input convex neural networks (ICNNs)

In input convex neural networks, the neuron weights are constrained to be nonnegative and weighted skip connections are added from the input layer to each layer. More details can be found in [Amos et al. (2017)](literature.md). These changes make the network output convex with respect to the inputs. A convex piecewise linear function can be formulated as a linear programming problem (LP) which is much more computationally efficient than the MILP formulations of "regular" neural networks. This is the reason for implementing ICNN functionality into this package. ICNNs are a viable option when the data or function being modeled is approximately convex and/or some prediction accuracy must be sacrificed for computational performance.

## Training

The [Flux.jl](https://fluxml.ai/Flux.jl/stable/) interface doesn't allow for simple implementation of ICNNs as far as we know. However, the ICNN models can easily be implemented and trained using Tensorflow with Python, for example. The model parameters can then be exported as a JSON file and imported into Julia to create the LP formulation. An example on how to build the ICNN, train it, and export the parameters using the high-level Tensorflow interface can be found in the  `examples/`-folder of the [package repository](https://github.com/gamma-opt/Gogeta.jl). The requirements for the JSON file structure are listed in the function description of [`ICNN_incorporate!`](@ref).

## Formulation

The interface for formulating ICNNs as LPs has been designed to make incorporating them into a larger optimization problem, e.g. as surrogates, as easy as possible. The `JuMP` model is first built and the variables, constraints, and objective are added. The function [`ICNN_incorporate!`](@ref) takes as arguments the `JuMP` model, the relative filepath of the JSON file containing the ICNN parameters, the output variable, and finally the input variables (in order). Currently only models with one output variable are supported.

Build the model.

```julia
jump_model = Model(Gurobi.Optimizer)

@variable(jump_model, -1 <= x <= 1)
@variable(jump_model, -1 <= y <= 1)
@variable(jump_model, z)

@constraint(jump_model, y >= 1-x)

@objective(jump_model, Min, x+y)
```

Include input convex neural network as a part of the larger optimization problem.
The JSON file containing the ICNN parameters is called "model_weights.json".
The variables `x` and `y` are linked to the variable `z` by the ICNN.

```julia
ICNN_incorporate!(jump_model, "model_weights.json", z, x, y)

optimize!(jump_model)
solution_summary(jump_model)

# see optimal solution
value(x)
value(y)
value(z)
```

The problem is very fast to solve since no binary variables are added.

If one wants to use ICNNs "by themselves" for global optimization, for example, the same steps can be followed but without adding any extra variables or constraints.

Add input and output variables of the ICNN into the `JuMP` model and minimize the ICNN output as the objective.

```julia
jump_model = Model(Gurobi.Optimizer)

@variable(jump_model, -1 <= x <= 1)
@variable(jump_model, -1 <= y <= 1)
@variable(jump_model, z)

@objective(jump_model, Min, z)

ICNN_incorporate!(jump_model, "model_weights.json", z, x, y)

optimize!(jump_model)
solution_summary(jump_model)
```

## Considerations

### Feasibility

In an optimization problem where an ICNN has been incorporated as a surrogate, infeasibility might not be able to be detected. This is because the ICNN is formulated as an epigraph with a penalty added to the objective function minimizing the ICNN output. However, this penalty term doesn't prevent the solver from finding solutions that are "above" the ICNN function hypersurface. Therefore, the optimization problem structure should be studied carefully to reason whether it is possible to come up with these pseudo-feasible solutions. If studying the problem structure is too complex, the optimal solution  returned by the solver can be checked using the [`check_ICNN`](@ref) function. If the check fails, the optimization problem is likely infeasible but this has not been proved so the problem should be investigated more thoroughly.

### Penalty domination

The second important consideration is the objective function of the optimization problem where the ICNN has been incorporated as a surrogate. As stated in the previous paragrah, the ICNN LP formulation relies on a penalty term that is added to the objective function. Thus, if the objective function already includes a term which is linked to or is itself the ICNN output variable, the penalty term added in the [`ICNN_incorporate!`](@ref) will not have the desired effect of guaranteeing that the ICNN is satisfied. We have not figured out a way around this issue, so if the penalty term is "dominated" in the objective, ICNN surrogates are probably not suitable for the given optimization problem.
