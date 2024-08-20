# Neural networks in larger optimization problems

If a neural network is being used as a surrogate for a set of constraints in a separate optimization problem, for instance, its MILP formulation must be incorporated into that larger optimization problem in a way that links the necessary input and output variables. In addition, sometimes multiple different MILP surrogates must be added to an optimization problem to model multiple different constraints and therefore variable names cannot be duplicated. In this case using [`NN_formulate!`](@ref) will not be sufficient as it adds named variables. [`NN_incorporate!`](@ref) satisfies the surrogate modeling requirements by adding the MILP formulation of a given neural network to the given `JuMP` model by using anonymous variables and linking the neural surrogate input and output variables to the desired variables in the `JuMP` model.


## Formulation

The variable to be linked to the output of the neural network is given as the third argument to the function, and the variables to be linked to the input variables are given last, in order.

```julia
init_U = [-0.5, 0.5];
init_L = [-1.5, -0.5];

# create an optimization problem
jump_model = Model(Gurobi.Optimizer)

@variable(jump_model, -1.5 <= x <= -0.5)
@variable(jump_model, -0.5 <= y <= 0.5)
@variable(jump_model, output)

@constraint(jump_model, y >= -x - 1)

@objective(jump_model, Max, output - 0.5*x)

NN_incorporate!(jump_model, NN_model, output, x, y; U_in=init_U, L_in=init_L)
```

[`NN_incorporate!`](@ref) contains the same compression and bound tightening options as [`NN_formulate!`](@ref) as keyword arguments. If optimization-based bound tightening is used, a separate `set_solver!` function must be defined in the global scope because a separate `JuMP` model is created and used for calculating the bounds.


```julia
ENV = Gurobi.Env()
function set_solver!(jump)
    set_optimizer(jump, () -> Gurobi.Optimizer(ENV))
    #relax_integrality(jump) # Use this to solve bounds with binary variables relaxed. Looser bounds but faster bound tightening.
    set_silent(jump)
end
```

!!! note
    
    When using `Gurobi` as the solver, the environment should be defined once and saved to prevent new environments being created for each of the bound calculations.

## Tensorflow models

In addition to `Flux.Chain` neural networks, [`NN_incorporate!`](@ref) also accepts parameter sets from e.g. Tensorflow models as input. This lets the user train the neural networks with his preferred libraries as long as the trained parameters are exported as a JSON file. The JSON file structure requirements are listed in the function description of [`NN_incorporate!`](@ref). The relative location of the JSON file is given to the function as an argument in the same position where the `Flux` model would be.

```julia
NN_incorporate!(jump_model, "folder/parameters.json", output, x, y; U_in=init_U, L_in=init_L)
```

Where "folder/parameters.json" is the relative path of the JSON file containing the neural network parameters.

In the next section, we look at the special case of the neural networks.