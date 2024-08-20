# Formulation of NN with big-M approach

The first way to formulate NN as a MIP is to use function [`NN_formulate!`](@ref). This formulation is based on the following paper: [Fischetti and Jo (2018)](literature.md). For more detailed information with examples, please see next [jupyter notebook](https://github.com/gamma-opt/Gogeta.jl/blob/main/examples/neural_networks/example_1_neural_networks.ipynb).

Suppose you have a trained neural network `NN_model` with known boundaries for input variables (`init_U`, `init_L`), then a trained NN can be formulated as `JuMP` model:

```julia
using Flux
NN_model = Chain(
    Dense(2 => 10, relu),
    Dense(10 => 20, relu),
    Dense(20 => 5, relu),
    Dense(5 => 1)
)

init_U = [-0.5, 0.5];
init_L = [-1.5, -0.5];

jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model) # set desired parameters
bounds_U, bounds_L = NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="standard", compress=true)
```

The function returns boundaries for each neuron and the `jump_model` is updated in the function. By default objective function of the `jump_model` is set to the dummy function *"Max 1"*.

This formulation enables compression by setting `compress=true`. Compression drops inactive neurons (or dead neurons) and decreases size of the MILP.

Possible bound-tightening strategies include: `fast` (default), `standard`, `output`, and `precomputed`.

!!! note

    When you use `precomputed` bound-tightening, you should also provide upper and loswer boundaries for the neurons (`U_bounds`, `L_bounds`) and nothing is returned.

```julia
 NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="precomputed", U_bounds=bounds_U, L_bounds=bounds_L, compress=true)
```

!!! note

    When you use `output` bound-tightening, you should also provide boundaries for the output neuron and nothing is returned.

```julia
 NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="output", U_out=U_out, L_out=L_out, compress=true)
```
## Compression of the NN using bounds 

Given lower and upper bounds (`bounds_U`, `bounds_L`) for neurons, the NN can be compressed. The function [`NN_compress`](@ref) will return modified compressed NN along with indexes of dropped neurons.

```julia
compressed, removed = NN_compress(NN_model, init_U, init_L, bounds_U, bounds_L)
```

## Calculation of the formulation output

When you have a ready formulation of the neural network, you can calculate the output of `JuMP` model with a function [`forward_pass!`](@ref)

```julia
forward_pass!(jump_model, [-1.0, 0.0])
```
## Running the formulation in parallel

!!! tip

    If formulation with `standard` bound-tightening takes too slow, you can reduce computation time by running formulation in parallel. For this you need to innitialize 'workers' and set `parallel = true`.  See next [jupyter notebook](https://github.com/gamma-opt/Gogeta.jl/tree/main/examples/neural_networks/example_2_neural_networks_parallel) for a more detailed explanation.

```julia
# Create the workers
using Distributed
addprocs(4)
@everywhere using Gurobi

# In order to prevent Gurobi obtaining a new license for each solve
@everywhere ENV = Ref{Gurobi.Env}()

@everywhere function init_env()
    global ENV
    ENV[] = Gurobi.Env()
end

for worker in workers()
    fetch(@spawnat worker init_env())
end

# Regardless of the solver, this must be defined
@everywhere using JuMP

@everywhere function set_solver!(jump_model)
    set_optimizer(jump_model, () -> Gurobi.Optimizer(ENV[]))
    set_silent(jump_model)
end

@everywhere using Gogeta

# Create a JuMP model from the neural network with parallel bound tightening.
jump = NN_model()
@time U, L = NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="standard", silent=false, parallel=true);
```
In the next section, we will look at the Psplit formulation of NNs.