# Formulation of CNNs

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