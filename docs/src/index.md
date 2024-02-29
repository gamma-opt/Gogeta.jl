# Gogeta.jl

[Gogeta](https://gamma-opt.github.io/Gogeta.jl/) is a package that enables the user to formulate machine-learning models as mathematical programming problems.

Currently supported models are `Flux.Chain` ReLU-activated neural networks (dense and convolutional) and `EvoTrees` tree ensemble models.

## Installation
```julia-repl
julia> Pkg.add("Gogeta")
```

## Getting started

### Tree ensembles

First, one must create and train an `EvoTrees` tree ensemble model.

```julia
using EvoTrees

config = EvoTreeRegressor(nrounds=500, max_depth=5);
evo_model = fit_evotree(config; x_train, y_train);
```

Then the parameters can be extracted from the trained tree ensemble and used to create a `JuMP` model containing the tree ensemble MIP formulation.

```julia
using Gurobi
using Gogeta

# Extract data from EvoTrees model

universal_tree_model = extract_evotrees_info(evo_model);

# Create JuMP model
opt_model = TE_to_MIP(universal_tree_model, Gurobi.Optimizer(), MIN_SENSE);
```

There are two ways of optimizing the JuMP model: either by 1) creating the full set of split constraints before optimizing, or 2) using lazy constraints to generate only the necessary ones during the solution process.

```julia
optimize_with_lazy_constraints!(opt_model, universal_tree_model)
```

```julia
optimize_with_initial_constraints!(opt_model, universal_tree_model)
```

The optimal solution (minimum and maximum values for each of the input variables) can be queried after the optimization.

```julia
get_solution(opt_model, universal_tree_model)
objective_value(opt_model)
```

### Neural networks

With neural networks, the hidden layers must use the $ReLU$ activation function, and the output layer must use the identity activation.

These neural networks can be formulated into mixed-integer optimization problems. 
Along with formulation, the neuron activation bounds can be calculated, which improves computational performance as well as enables compression.

The network is compressed by pruning neurons that are either stabily active or inactive. The activation bounds are used to identify these neurons.

First, create a neural network model satisfying the requirements:

```julia
using Flux

model = Chain(
    Dense(2 => 10, relu),
    Dense(10 => 20, relu),
    Dense(20 => 5, relu),
    Dense(5 => 1)
)
```

Then define the bounds for the input variables. These will be used to calculate the activation bounds for the subsequent layers.

```julia
init_U = [-0.5, 0.5];
init_L = [-1.5, -0.5];
```

One needs to provide parameters for the solver. In this case `Gurobi` solver is used with console logging off, default thread count, linear relaxation off and infinite time limit.

```julia
solver_params = SolverParams(solver="Gurobi", silent=true, threads=0, relax=false, time_limit=0);
```

Then the neural network can be formulated into a MIP. Here bound tightening is also used.

```julia
nn_jump, U_correct, L_correct = NN_to_MIP_with_bound_tightening(model, init_U, init_L, solver_params; bound_tightening="standard");
```

Using these bounds, the model can be compressed.

```julia
compressed, removed = compress_with_precomputed(model, init_U, init_L, U_correct, L_correct);
```

Compression can also be done without precomputed bounds.

```julia
jump_model, compressed_model, removed_neurons, bounds_U, bounds_L = compress_with_bound_tightening(model, init_U, init_L,solver_params; bound_tightening="standard");
```

Use the `JuMP` model to calculate a forward pass through the network (input at the center of the domain).

```julia
forward_pass!(jump_model, [-1.0, 0.0])
```

#### Convolutional neural networks

The convolutional neural network requirements can be found in the [`create_MIP_from_CNN!`](@ref) documentation.

First, create some kind of input (or load an image from your computer).

```julia
input = rand(Float32, 70, 50, 1, 1) # BW 70x50 image
```

Then, create a convolutional neural network model satisfying the requirements:

```julia
using Flux

CNN_model = Flux.Chain(
    Conv((4,3), 1 => 10, pad=(2, 1), stride=(3, 2), relu),
    MeanPool((5,3), pad=(3, 2), stride=(2, 2)),
    MaxPool((3,4), pad=(1, 3), stride=(3, 2)),
    Conv((4,3), 10 => 5, pad=(2, 1), stride=(3, 2), relu),
    MaxPool((3,4), pad=(1, 3), stride=(3, 2)),
    Flux.flatten,
    Dense(20 => 100, relu),
    Dense(100 => 1)
)
```

Then, create an empty `JuMP` model, extract the layer structure of the CNN model and finally formulate the MIP.

```julia
jump = Model(Gurobi.Optimizer)
set_silent(jump)
cnns = get_structure(CNN_model, input);
create_MIP_from_CNN!(jump, CNN_model, cnns)
```

Check that the `JuMP` model produces the same outputs as the `Flux.Chain`.

```julia
vec(CNN_model(input)) ≈ image_pass!(jump, input)
```

## How to use?

Using the tree ensemble optimization from this package is quite straightforward. The only parameter the user can change is the solution method: with initial constraints or with lazy constraints.
In our computational tests, we have seen that the lazy constraint generation almost invariably produces models that are computationally easier to solve. 
Therefore we recommend primarily using it as the solution method, but depending on your use case, trying the initial constraints might also be worthwhile.

Conversely, the choice of the best neural network bound tightening and compression procedures depends heavily on your specific use case. 
Based on some limited computational tests of our own as well knowledge from the field, we can make the following general recommendations:

* Wide but shallow neural networks should be preferred. The bound tightening gets exponentially harder with deeper layers.
* For small neural network models, using the "fast" bound tightening option is probably the best, since the resulting formulations are easy to solve even with loose bounds.
* For larger neural networks, "standard" bound tightening will produce tighter bounds but take more time. However, when using the `JuMP` model, the tighter bounds might make it more computationally feasible.
* For large neural networks where the output bounds are known, "output" bound tightening can be used. This bound tightening is very slow but might be necessary to increase the computational feasibility of the resulting `JuMP` model.
* If the model has many so-called "dead" neurons, creating the JuMP model by using compression is beneficial, since the formulation will have fewer constraints and the bound tightening will be faster, reducing total formulation time.

These are only general recommendations based on limited evidence, and the user should validate the performance of each bound tightening and compression procedure in relation to her own work.