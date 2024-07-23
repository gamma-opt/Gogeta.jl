# Neural networks

With neural networks, the hidden layers must use the $ReLU$ activation function, and the output layer must use the identity activation.

A neural networks satifying these requirements can be formulated into a mixed-integer optimization problem. 
Along with formulation, the neuron activation bounds can be calculated, which improves computational performance as well as enables compression.

The network is compressed by pruning neurons that are either stabily active or inactive. The activation bounds are used to identify these neurons.

## Formulation

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

Now the neural network can be formulated into a MIP. Here optimization-based bound tightening is also used.

```julia
jump_model = Model(Gurobi.Optimizer)
set_silent(model) # set desired parameters

bounds_U, bounds_L = NN_formulate!(jump_model, model, init_U, init_L; bound_tightening="standard")
```

Using these bounds, the model can be compressed.

```julia
compressed, removed = NN_compress(model, init_U, init_L, bounds_U, bounds_L)
```

Compression can also be done without precomputed bounds.

```julia
bounds_U, bounds_L = NN_formulate!(jump_model, model, init_U, init_L; bound_tightening="standard", compress=true)
```

Use the `JuMP` model to calculate a forward pass through the network (input at the center of the domain).

```julia
forward_pass!(jump_model, [-1.0, 0.0])
```

Finding the optimum of the neural network is now very straightforward.

```julia
last_layer, _ = maximum(keys(jump_model[:x].data))
@objective(jump_model, Max, jump_model[:x][last_layer, 1]) # maximize the output neuron
optimize!(jump_model)
value.(jump_model[:x][0, :]) # maximum
objective_value(jump_model) # neural network output at maximum
```

## Convolutional neural networks

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

## Bound tightening

To improve the computational feasibility of the mixed-integer formulation of a neural network, the big-M values associated with the some of the constraints can be made smaller by calculating the minimum and and maximum activations of the individual neurons. This can be done with a heuristic algorithm or by using mathematical optimization.

Our package includes three different modes of bound tightening: `fast` *(default)*, `standard` and `output`.
1. The `fast` mode uses a heuristic algorithm to determine the neuron activation bounds only based on the activation bounds of the previous layer. This algorithm practically doesn't increase the formulation time, so it is enabled by default. 
2. The `standard` mode considers the whole mixed-integer problem with variables and constraints defined up to the previous layer from the neuron under bound tightening. It uses optimization to find the neuron activation bounds and is therefore significantly slower than the `fast` mode but is able to produce tighter bounds (smaller big-M values).
3. In some situations, the user might know the bounds for the output layer neurons. The `output` mode takes into account these output bounds as well as the whole MIP. Therefore, it is able to produce the tightest bounds of all the methods listed, but it is also the slowest.

(`precomputed` is also one of the bound tightening options in the functions. It can be used by inputting bounds that have already been calculated.)

A detailed discussion on bound tightening techniques can be found in [Grimstad and Andresson (2019)](literature.md).

## Sampling

Instead of just solving the MIP, the neural network can be optimized (finding the output maximizing/minimizing input) by using a sampling approach. Note that these features are experimental and cannot be guranteed to find the global optimum.

```julia
using QuasiMonteCarlo

jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model)
NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="fast");

# set objective function as the last layer output
last_layer, _ = maximum(keys(jump_model[:x].data))
@objective(jump_model, Max, jump_model[:x][last_layer, 1])

samples = QuasiMonteCarlo.sample(1000, init_L, init_U, LatinHypercubeSample());
x_opt, optimum = optimize_by_sampling!(jump_model, samples);
```

### Relaxing walk algorithm

Another method for heuristically optimizing the JuMP model is the so-called relaxing walk algorithm. It is based on a sampling approach that utilizes LP relaxations of the original problem and a pseudo gradient descent -algorithm.

```julia
jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model)
NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="fast")

# set objective function as the last layer output
last_layer, _ = maximum(keys(jump_model[:x].data))
@objective(jump_model, Max, jump_model[:x][last_layer, 1])

x_opt, optimum = optimize_by_walking!(jump_model, init_U, init_L)
```

A `set_solver!` -function must be specified (used for copying the model in the algorithm).

```julia
function set_solver!(jump)
    set_optimizer(jump, Gurobi.Optimizer)
    set_silent(jump)
end
```

## NN with partitions

The formulation of NN is done with function `NN_formulate_Psplit!`. It has next set of input parameters</br>
<ul>
<li><code>jump_model</code>: The constraints and variables will be saved to this optimization model.</li>
<li><code>NN_model</code>: Neural network model to be formulated.</li>
<li><code>P</code>: The number of splits</li>
<li><code>U_in</code>: Upper bounds for the input variables.</li>
<li><code>L_in</code>: Lower bounds for the input variables.</li>
<li> <code>strategy</code> (optional): the way partitioning is done, possible options include: "equalsize", "equalrange", "random". Default is "equalsize".</li>
<li> <code>silent</code>(optional): Controls console ouput.</li>
<li><code>bound_tightening</code> (optional): possible options include: "standart", "precomputed", and "fast" (default)</li>
</ul>

## Recommendations

The choice of the best neural network bound tightening and compression procedures depends heavily on your specific use case. 
Based on some limited computational tests of our own as well as knowledge from the field, we can make the following general recommendations:

* Wide but shallow neural networks should be preferred. The bound tightening gets exponentially harder with deeper layers.
* For small neural network models, using the "fast" bound tightening option is probably the best, since the resulting formulations are easy to solve even with loose bounds.
* For larger neural networks, "standard" bound tightening will produce tighter bounds but take more time. However, when using the `JuMP` model, the tighter bounds might make it more computationally feasible.
* For large neural networks where the output bounds are known, "output" bound tightening can be used. This bound tightening is very slow but might be necessary to increase the computational feasibility of the resulting `JuMP` model.
* If the model has many so-called "dead" neurons, creating the JuMP model by using compression is beneficial, since the formulation will have fewer constraints and the bound tightening will be faster, reducing total formulation time.

These are only general recommendations based on limited evidence, and the user should validate the performance of each bound tightening and compression procedure in relation to her own work.