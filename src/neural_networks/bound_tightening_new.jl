using BSON
using Flux
using JuMP

BSON.@load string(@__DIR__)*"/NN_paraboloid.bson" model

NN_model = model

init_ub = [1.0f0, 1.0f0]
init_lb = [-1.0f0, -1.0f0]

# Function begins

K = length(NN_model) # number of layers (input layer not included)
W = [Flux.params(NN_model)[2*k-1] for k in 1:K]
b = [Flux.params(NN_model)[2*k] for k in 1:K]

input_length = Int((length(W[1]) / length(b[1])))
neuron_count = [length(b[k]) for k in eachindex(b)]

neurons(layer) = layer == 0 ? [i for i in 1:input_length] : [i for i in 1:neuron_count[layer]]

@assert input_length == length(init_ub) == length(init_lb) "Initial bounds arrays must be the same length as the input layer"

# build model up to second layer
jump_model = JuMP.Model()

@variable(jump_model, x[layer = 0:K, neurons(layer)])
@variable(jump_model, s[layer = 0:K, neurons(layer)])
@variable(jump_model, z[layer = 0:K, neurons(layer)])

@constraint(jump_model, [j = 1:input_length], init_lb[j] <= x[0, j] <= init_ub[j])

for layer in 1:K # hidden layers and output layer - second layer and up

    ub_x = []
    ub_s = []

    # TODO: the model must be copied for each neuron in a new layer

    for neuron in 1:neuron_count[layer]

        @constraint(jump_model, x[layer, neuron] >= 0)
        @constraint(jump_model, s[layer, neuron] >= 0)
        set_binary(z[layer, neuron])

        @constraint(jump_model, z[layer, neuron] --> {x[layer, neuron] <= 0})
        @constraint(jump_model, !z[layer, neuron] --> {s[layer, neuron] <= 0})

        @constraint(jump_model, x[layer, neuron] - s[layer, neuron] == b[layer][neuron] + sum(W[layer][neuron, i] * x[layer-1, i] for i in neurons(layer-1)))

        @objective(jump_model, Max, x[layer, neuron]) # ub_x
        @objective(jump_model, Max, s[layer, neuron]) # ub_s
    end

    # add lower and upper bound constraints for x and s 
end

return jump_model

function bound_tightening(NN_model::Flux.Chain, init_ub::Vector{Float64}, init_lb::Vector{Float64})

end