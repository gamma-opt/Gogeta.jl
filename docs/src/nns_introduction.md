# Neural Networks

With the `Gogeta` package, it is currently possible to formulate deep neural networks (NNs) and convolutional neural networks (CNNs) as mixed integer problems (MIP). We start the discussion from NNs. More detailed code can be found in the [example jupyter notebooks](https://github.com/gamma-opt/Gogeta.jl/tree/main/examples).

Currently, there are two different ways how to formulate `Flux.Chain` neural network models as MILPs: using the [big-M approach](neural_networks.md) or the [Psplits](psplit_nns.md) formulation. In this section, we are going to introduce some concepts related to both of them. 

!!! note

    This and following sections describe how to formulate a `Flux.Chain` neural network model as a MILP. If you want to use neural networks as surrogate models in a larger optimization problem, [this section](nns_in_larger.md) has a guide on how to accomplish this effectively and formulate the neural network with anonymous variables.

## Requirements for the architecture of the NN

In order to formulate NNs as MIPs, they must use the $ReLU$ activation function at each hidden layer, and the output layer must use the identity activation function. See the example below:

```julia
using Flux

NN_model = Chain(
    Dense(2 => 10, relu),
    Dense(10 => 20, relu),
    Dense(20 => 5, relu),
    Dense(5 => 1)
)
```
For each input variable, upper and lower bounds need to be provided. The formulation will ensure to produce the same output as NN in these ranges.

```julia
init_U = [-0.5, 0.5];
init_L = [-1.5, -0.5];
```

A neural network satifying these requirements can be formulated into a mixed-integer linear optimization problem (MILP). 
Along with formulation, the neuron activation bounds can be calculated, which improves computational performance as well as enables compression.

## Bound-tightening options

The formulations require calculating boundary values for each neuron. `bound_tightening` refers to the way how the boundaries are calculated.

The *(default)* way to do this is called `fast`, but you may also use `standard`,`precomputed`, `output`.

1. The `fast` mode uses a heuristic algorithm to determine the neuron activation bounds only based on the activation bounds of the previous layer. This algorithm practically doesn't increase the formulation time, so it is enabled by default. 
2. The `standard` mode considers the whole mixed-integer problem with variables and constraints defined up to the previous layer from the neuron under bound tightening. It uses optimization to find the neuron activation bounds and is therefore significantly slower than the `fast` mode but is able to produce tighter bounds (smaller big-M values).
3. In some situations, the user might know the bounds for the output layer neurons. The `output` mode takes into account these output bounds as well as the whole MIP. Therefore, it is able to produce the tightest bounds of all the methods listed, but it is also the slowest.
4. `precomputed` is the last of the bound tightening options. It can be used to input bounds that have already been calculated.

A detailed discussion on bound tightening techniques can be found in [Grimstad and Andresson (2019)](literature.md).

## Recommendations

The choice of the best neural network bound tightening and compression procedures depends heavily on your specific use case. 
Based on some limited computational tests of our own as well as knowledge from the field, we can make the following general recommendations:

* Wide but shallow neural networks should be preferred. The bound tightening gets exponentially harder with deeper layers.
* For small neural network models, using the `fast` bound tightening option is probably the best, since the resulting formulations are easy to solve even with loose bounds.
* For larger neural networks, `standard` bound tightening will produce tighter bounds but take more time. However, when using the `JuMP` model, the tighter bounds might make it more computationally feasible.
* For large neural networks where the output bounds are known, `output` bound tightening can be used. This bound tightening is very slow but might be necessary to increase the computational feasibility of the resulting `JuMP` model.
* If the model has many so-called "dead" neurons, creating the `JuMP` model by using compression is beneficial since the formulation will have fewer constraints and the bound tightening will be faster, reducing total formulation time.
* With the partition-based formulation, choose a number of partitions wisely, since it greatly increases size of the MIP problem. You are interested in chosing the minimum number of partitions that result in the tightest bounds for the neurons.

These are only general recommendations based on limited evidence, and the user should validate the performance of each bound tightening and compression procedure in relation to her own work.