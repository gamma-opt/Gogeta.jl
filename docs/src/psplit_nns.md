# Formulation of NN with the partition-based approach

We implemented partition-based (or Psplit) approach based on the paper written by [Tsay et al. (2021)](literature.md). The idea behind this approach is to split weights of neurons into non-overlapping partitions while formulation. A more detailed example can be found in the next [juputer notebook](https://github.com/gamma-opt/Gogeta.jl/blob/main/examples/neural_networks/example_5_nn_partition_formulation.ipynb).

In order to formulate NN as `JuMP` model, you should call [`NN_formulate_Psplit!`](@ref). It has parameter `P` that controls how many partitions are created.

!!! tip

    Always select a reasonable number of partitions, since too big number `P` can result in generation of empty partitions. 

## Partition strategies

With Psplit approach, you can also select a `strategy` how the partitions are formulated. There are four different partition strategies available: `equalsize` (default), `equalrange`, `snake`, `random`.
* `equalsize` – Sorts weights and splits them into $P$ non-overlapping sets of the same size in order
* `equalrange` – Sorts weights, puts all weights less than 5% percentile of data to the first partition and all weigths more than 95% percentile into the last partition. All sets are ensured to have the same range of weights. The minimum number of partitions is 3
* `random` – Randomly assigns weights to $P$ sets
* `snake` – Sorts the weights and assigns weights to sets in snake order

!!! tip

    Authors of the paper advise to use  `equalsize`  and `equalrange` partitions strategies since they ensure that weights in the sets would be of relatively the same order. `random` and `snake` are expected to perform much worse and are used for comparison purposes. Though these assumptions are true only in case `P`<< `N` (where N is the number of neurons)

## Important matters

* For parttion based approach there are several `bound_tightening` strategies available: `fast` (default), `standard`, `precomputed`.
* Compression is not yet available.
* The function can be also runned in parallel in the same was as in [Big-M formulation](neural_networks.md). 
* Requirements for the neural network are exactly the same. 
* The output of the formulation can be calculated using function [`forward_pass!`](@ref) as  in [Big-M formulation](neural_networks.md).

## Example

To get started, you can copy this piece of code and try changing number of partitions `P`, `bound_tigtening`, `parallel` and `strategy` parametres.


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
P = 3
bounds_U, bounds_L = bounds_U_st, bounds_L_st = NN_formulate_Psplit!(jump_model, NN_model, P, init_U, init_L, bound_tightening="standard", strategy="equalrange");
```

In the next section, you will understand how the MILP formulations can be optimized with `Gogeta`.