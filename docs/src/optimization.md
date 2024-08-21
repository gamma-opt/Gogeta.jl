# Finding the optimum of the neural network

Being able to formulate NNs as a MIPs gives us the possibility to optimize over "black box" models. For instance, imagine that we have a dataset with some information about houses and their associated prices. With a NN trained on this dataset and its MIP formulation, we can find what kind of houses would be the most affordable and the most expensive. We could also add some additional constraints to the "target" house and see what is the possible range of prices.

In this section, we will optimize over the output neuron of NNs, but you can choose any objective function. The variable associated with the output neuron can be extracted in the following way:

```julia
output_neuron = jump_model[:x][maximum(keys(jump_model[:x].data))]
@objective(jump_model, Max, output_neuron) # maximize the output neuron
```

## Direct optimization

We can optimize the model directly.

```julia
optimize!(jump_model)
value.(jump_model[:x][0, :]) # maximum
objective_value(jump_model) # neural network output at maximum
```

## Sampling

!!! warning
    This method works only with the Big-M formulation

Instead of just solving the MILP, the neural network can be optimized (finding the output maximizing/minimizing input) by using a sampling approach. See [jupyter notebook](https://github.com/gamma-opt/Gogeta.jl/blob/main/examples/neural_networks/example_4_nn_relaxing_walk.ipynb) for a more detailed example.

!!! note
    Much more effective algorithms for finding the optimum of a trained neural network exist, such as projected gradient descent. The sampling-based optimization algorithms implemented in this package are best intended for satisfying one's curiosity and understanding the problem structure better.

First we formulate the NN as a MIP

```julia
using QuasiMonteCarlo

jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model)
NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="fast");
```
Then, we set objective function to either minimize or maximize the output neuron.

```julia
# set objective function as the last layer output
output_neuron = jump_model[:x][maximum(keys(jump_model[:x].data))]
@objective(jump_model, Max, output_neuron)
```
Randomly generate samples that aling with lower and upper bounds. Call function `optimize_by_sampling!` that returns nearly optimum solution.
```julia
samples = QuasiMonteCarlo.sample(1000, init_L, init_U, LatinHypercubeSample());
x_opt, optimum = optimize_by_sampling!(jump_model, samples);
```
## Relaxing walk algorithm

Another method for heuristically optimizing the JuMP model is the so-called relaxing walk algorithm. It is based on a sampling approach that utilizes LP relaxations of the original problem and a pseudo gradient descent -algorithm. It uses function [`optimize_by_walking!`](@ref). See [jupyter notebook](https://github.com/gamma-opt/Gogeta.jl/blob/main/examples/neural_networks/example_3_nn_sampling.ipynb) for a more detailed example.

!!! warning
    This method works only with Big-M formulation of NN

```julia
jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model)
NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="fast")
# set objective function as the last layer output
output_neuron = jump_model[:x][maximum(keys(jump_model[:x].data))]
@objective(jump_model, Max, output_neuron)
x_opt, optimum = optimize_by_walking!(jump_model, NN_model, init_U, init_L)
```

A `set_solver!` - function must be specified (used for copying the model in the algorithm). This function is different depending on the optimizer.

```julia
function set_solver!(jump)
    set_optimizer(jump, Gurobi.Optimizer)
    set_silent(jump)
end
```


