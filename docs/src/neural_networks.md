# Neural networks
With `Gogeta` library it is currently possible to formulate deep neural networks (NNs) and convolutional neural networks (CNNs) as a mixed integer problems (MIP). We would start discussion from the NNs. More detailed discussion on each formulation can be found in the [example jupyter notebooks](https://github.com/gamma-opt/Gogeta.jl/tree/main/examples).

## Formulation of NNs

Being able to formulate NN as a MIP gives us posibility to optimize over "black box" model. Consider that we have a dataset with some information about housing prices. With a NN trained on this dataset and MIP formulation of it, we can find housing with the minimum and maximum prices. We could also add some additional constraints to the "target" housing and see what is a possible range of prices. 

### Requirements for the architecture of NN

Currently, we are able to formulate NNs use the $ReLU$ activation function at each hidden layer, and the output layer must use the identity activation. See the example below:

```julia
using Flux

NN_model = Chain(
    Dense(2 => 10, relu),
    Dense(10 => 20, relu),
    Dense(20 => 5, relu),
    Dense(5 => 1)
)
```
For each input variable, you should also provide upper and lower bounds. The formulation will ensure to produce the same output as NN in these ranges.

```julia
init_U = [-0.5, 0.5];
init_L = [-1.5, -0.5];
```

### Bound-tightening options

Before proceeding to formulations of NNs, we should introduce some definitions. The formulations require to calculate boundary values for each neuron. `bound_tightening` refers to the way how the boundaries are calculated.

The *(default)* way to do this is called `fast`, but you may also face with `standard`,`precomputed`, `output`.

1. The `fast` mode uses a heuristic algorithm to determine the neuron activation bounds only based on the activation bounds of the previous layer. This algorithm practically doesn't increase the formulation time, so it is enabled by default. 
2. The `standard` mode considers the whole mixed-integer problem with variables and constraints defined up to the previous layer from the neuron under bound tightening. It uses optimization to find the neuron activation bounds and is therefore significantly slower than the `fast` mode but is able to produce tighter bounds (smaller big-M values).
3. In some situations, the user might know the bounds for the output layer neurons. The `output` mode takes into account these output bounds as well as the whole MIP. Therefore, it is able to produce the tightest bounds of all the methods listed, but it is also the slowest.
4. `precomputed` is also one of the bound tightening options in the functions. It can be used by inputting bounds that have already been calculated.

A detailed discussion on bound tightening techniques can be found in [Grimstad and Andresson (2019)](literature.md).


### Formulation of NN with big-M approach
The first way to formulate NN as a MIP is to use function [`NN_formulate!`](@ref). This formulation is based on the following paper: [Fischetti and Jo (2018)](literature.md).

This formulation enables compression by setting `compress=true`, running bound-tightening in parallel (only used in `standard` bound-tightening). Possible bound-tightening strategies include: `fast` (default), `standard`, `output`, `precomputed`.
#### Example
```julia
jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model) # set desired parameters
bounds_U, bounds_L = NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="standard", compress=true, parallel=true)
```
The function returns boundaries for each neuron, the `jump_model` is updated in the function. By default objective function of the `jump_model` is set to dummy function *"Max 1"*.

***Note:*** When you use `precomputed` bound-tightening, you should also provide boundaries for the neurons and nothing is returned.

```julia
 NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="precomputed", U_bounds=bounds_U, L_bounds=bounds_L compress=true, parallel=true)
```

### Formulation of NN with partition-based approach

The partition-based approach implemented in our library is based on the paper written by [Tsay et al. (2021)](literature.md). The idea behind this approach is to split weigth into non-overlapping partitions while formulation. 

#### Partition strategies
There are four different partition strategies availble: `equalsize` (default), `equalrange`, `snake`, `random`.
<ul>
<li><code>equalsize</code> Sorts weight and split them into $P$ non-overlaping sets of the same size in order</li>
<li><code>equalrange</code>Sorts weight, puts all weigths less than 5% percentile of data to the first partition and all weigths more than 95% percentile into the last partition. All sets are ensured to have the same range of weights. Minimum number of partitoons is 3</li>
<li><code>random</code>Randomly assigns weights to $P$ sets</li>
<li><code>snake</code> Sorts the weights and assigns weights to sets in snake order</li>
</ul>

It is adviced to use  `equalsize`  and `equalrange` partitions strategies, since they ensure that weights in the sets would be of relatively the same order. `random` and `snake` are proposed to be oposing strategies to these to and expecte to perform much worse.

#### Example
```julia
jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model) # set desired parameters
P = 3
bounds_U, bounds_L = bounds_U_st, bounds_L_st = NN_formulate_Psplit!(jump_model, NN_model, P, init_U, init_L, bound_tightening="standard", strategy="equalrange");
```

### Calculation of the formulation output

When you have a ready formulation of the neural network, you can calculate the output of `JuMP` model with a function [`forward_pass!`](@ref)

```julia
forward_pass!(jump_model, [-1.0, 0.0])
```
### Finding optimum of the neural network

It is quite straightforward. We just need to select the output neuron (neuron at the last layer) and either minimize or maximize it.

```julia
last_layer, _ = maximum(keys(jump_model[:x].data))
@objective(jump_model, Max, jump_model[:x][last_layer, 1]) # maximize the output neuron
optimize!(jump_model)
value.(jump_model[:x][0, :]) # maximum
objective_value(jump_model) # neural network output at maximum
```

### Compression of the NN using bounds 

Given bounds for neurons, the NN can be compressed. The function [`NN_compress`](@ref) will return modified compressed NN along with indexes of dropped neurons.

```julia
compressed, removed = NN_compress(NN_model, init_U, init_L, bounds_U, bounds_L)
```

### Additional features available
These two features can be used for formulations with big-M approach only. 

#### Sampling
Instead of just solving the MIP, the neural network can be optimized (finding the output maximizing/minimizing input) by using a sampling approach. Note that these features are experimental and cannot be guranteed to find the global optimum.

At first we formulate NN as a MIP

```julia
using QuasiMonteCarlo

jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model)
NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="fast");
```
Then, we set objective function to either minimizing or maximizing of the output neuron.

```julia
# set objective function as the last layer output
last_layer, _ = maximum(keys(jump_model[:x].data))
@objective(jump_model, Max, jump_model[:x][last_layer, 1])
```
Randomly generate samples that aling with lower and upper bounds. Call function `optimize_by_sampling!` that return nearly optimum solution.
```julia
samples = QuasiMonteCarlo.sample(1000, init_L, init_U, LatinHypercubeSample());
x_opt, optimum = optimize_by_sampling!(jump_model, samples);
```
#### Relaxing walk algorithm

Another method for heuristically optimizing the JuMP model is the so-called relaxing walk algorithm. It is based on a sampling approach that utilizes LP relaxations of the original problem and a pseudo gradient descent -algorithm. It uses function [`optimize_by_walking!`](@ref)

```julia
jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model)
NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="fast")
# set objective function as the last layer output
last_layer, _ = maximum(keys(jump_model[:x].data))
@objective(jump_model, Max, jump_model[:x][last_layer, 1])
x_opt, optimum = optimize_by_walking!(jump_model, NN_model, init_U, init_L)
```

A `set_solver!` - function must be specified (used for copying the model in the algorithm). This function is different depending on the optimizer.

```julia
function set_solver!(jump)
    set_optimizer(jump, Gurobi.Optimizer)
    set_silent(jump)
end
```

## Formulation of CNNs

With our library, you can also formulate CNNs.
The convolutional neural network requirements can be found in the [`CNN_formulate!`](@ref) documentation.

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
CNN_formulate!(jump, CNN_model, cnns)
```

Check that the `JuMP` model produces the same outputs as the `Flux.Chain`.

```julia
vec(CNN_model(input)) ≈ image_pass!(jump, input)
```

## Recommendations

The choice of the best neural network bound tightening and compression procedures depends heavily on your specific use case. 
Based on some limited computational tests of our own as well as knowledge from the field, we can make the following general recommendations:

* Wide but shallow neural networks should be preferred. The bound tightening gets exponentially harder with deeper layers.
* For small neural network models, using the "fast" bound tightening option is probably the best, since the resulting formulations are easy to solve even with loose bounds.
* For larger neural networks, "standard" bound tightening will produce tighter bounds but take more time. However, when using the `JuMP` model, the tighter bounds might make it more computationally feasible.
* For large neural networks where the output bounds are known, "output" bound tightening can be used. This bound tightening is very slow but might be necessary to increase the computational feasibility of the resulting `JuMP` model.
* If the model has many so-called "dead" neurons, creating the JuMP model by using compression is beneficial, since the formulation will have fewer constraints and the bound tightening will be faster, reducing total formulation time.
* With partition based formulation, choose number of partitions wisely, since it greatly increases size of the MIP problem. You are interested in chosing the minimum number of partitions that result in the tightest bounds for the neurons.

These are only general recommendations based on limited evidence, and the user should validate the performance of each bound tightening and compression procedure in relation to her own work.