# Formulation of NNs with the big-M approach

Neural networks can be formulated as MIPs using the function [`NN_formulate!`](@ref). The formulation is based on the following paper: [Fischetti and Jo (2018)](literature.md). For more detailed information with examples, please see the [jupyter notebook](https://github.com/gamma-opt/Gogeta.jl/blob/main/examples/neural_networks/example_1_neural_networks.ipynb).

Assuming you have a trained neural network `NN_model` with known boundaries for input variables (`init_U`, `init_L`), a trained NN can be formulated as `JuMP` model:

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

The function returns boundaries for each neuron and the `jump_model` is updated by the function. By default, the objective function of the `jump_model` is set to the dummy function *"Max 1"*.

With this function, compression can be enabled by setting `compress=true`. Compression drops inactive neurons (dead neurons) and thus decreases size of the MILP.

Possible bound-tightening strategies include: `fast` (default), `standard`, `output`, and `precomputed`.

!!! note

    When you use `precomputed` bound-tightening, you should also provide upper and lower boundaries for the neurons (`U_bounds`, `L_bounds`) and nothing is returned.

```julia
 NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="precomputed", U_bounds=bounds_U, L_bounds=bounds_L, compress=true)
```

!!! note

    When you use `output` bound-tightening, you should also provide boundaries for the output neuron and nothing is returned.

```julia
 NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="output", U_out=U_out, L_out=L_out, compress=true)
```
## Compression of the NN using precomputed bounds 

Given lower and upper bounds (`bounds_U`, `bounds_L`) for neurons, the NN can be compressed. The function [`NN_compress`](@ref) will return the modified compressed NN along with indexes of dropped neurons.

```julia
compressed, removed = NN_compress(NN_model, init_U, init_L, bounds_U, bounds_L)
```

## Calculation of the model output

When you have a ready formulation of the neural network, you can calculate the output of `JuMP` model with the function [`forward_pass!`](@ref)

```julia
forward_pass!(jump_model, [-1.0, 0.0])
```
## Performing the formulation in parallel

!!! tip

    If formulation with `standard` bound-tightening is too slow, computational time can be reduced by running the formulation in parallel. For this, workers need to be initialized and `parallel`-argument set to true.  See the [jupyter notebook](https://github.com/gamma-opt/Gogeta.jl/tree/main/examples/neural_networks/example_2_neural_networks_parallel) for a more detailed explanation.

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

Here Gurobi is used. For other solvers this procedure might be simpler, since an environment doesn't have to be created for each of the workers.