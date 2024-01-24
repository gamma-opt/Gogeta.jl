using BSON
using Flux
using JuMP
using Gurobi

BSON.@load string(@__DIR__)*"/NN_paraboloid.bson" model

init_ub = [1.0, 1.0]
init_lb = [-1.0, -1.0]

model = Chain(
    Dense(2 => 30, relu),
    Dense(30 => 50, relu),
    Dense(50 => 1, relu)
) 

@time jump_model, bounds_x, bounds_s = bound_tightening(model, init_ub, init_lb)

data = rand(Float32, (2, 1000)) .- 0.5f0

x_train = data[:, 1:750]
y_train = [sum(x_train[:, col].^2) for col in 1:750]

vec(model(x_train)) ≈ [forward_pass!(jump_model, x_train[:, i])[1] for i in 1:750]

function bound_tightening(NN_model::Flux.Chain, init_ub::Vector{Float64}, init_lb::Vector{Float64})

    K = length(NN_model) # number of layers (input layer not included)
    W = [Flux.params(NN_model)[2*k-1] for k in 1:K]
    b = [Flux.params(NN_model)[2*k] for k in 1:K]
    
    input_length = Int((length(W[1]) / length(b[1])))
    neuron_count = [length(b[k]) for k in eachindex(b)]
    neurons(layer) = layer == 0 ? [i for i in 1:input_length] : [i for i in 1:neuron_count[layer]]
    
    @assert input_length == length(init_ub) == length(init_lb) "Initial bounds arrays must be the same length as the input layer"
    
    # build model up to second layer
    jump_model = direct_model(Gurobi.Optimizer())
    set_silent(jump_model)
    
    @variable(jump_model, x[layer = 0:K, neurons(layer)])
    @variable(jump_model, s[layer = 0:K, neurons(layer)])
    @variable(jump_model, z[layer = 0:K, neurons(layer)])
    
    @constraint(jump_model, [j = 1:input_length], x[0, j] <= init_ub[j])
    @constraint(jump_model, [j = 1:input_length], x[0, j] >= init_lb[j])
    
    bounds_x = Vector{Vector}(undef, K)
    bounds_s = Vector{Vector}(undef, K)
    
    for layer in 1:K # hidden layers to output layer - second layer and up

        println("LAYER $layer")
    
        ub_x = Vector{Float32}(undef, length(neurons(layer)))
        ub_s = Vector{Float32}(undef, length(neurons(layer)))
    
        # TODO: For parallelization the model must be copied for each neuron in a new layer to prevent data races
    
        for neuron in 1:neuron_count[layer]

            print("#")
    
            @constraint(jump_model, x[layer, neuron] >= 0)
            @constraint(jump_model, s[layer, neuron] >= 0)
            set_binary(z[layer, neuron])
    
            @constraint(jump_model, z[layer, neuron] --> {x[layer, neuron] <= 0})
            @constraint(jump_model, !z[layer, neuron] --> {s[layer, neuron] <= 0})
    
            @constraint(jump_model, x[layer, neuron] - s[layer, neuron] == b[layer][neuron] + sum(W[layer][neuron, i] * x[layer-1, i] for i in neurons(layer-1)))
    
            @objective(jump_model, Max, x[layer, neuron])
            optimize!(jump_model)
            ub_x[neuron] = objective_value(jump_model)
    
            @objective(jump_model, Max, s[layer, neuron])
            optimize!(jump_model)
            ub_s[neuron] = objective_value(jump_model)
        end

        println()
        
        bounds_x[layer] = ub_x
        bounds_s[layer] = ub_s
    
        @constraint(jump_model, [neuron = neurons(layer)], x[layer, neuron] <= ub_x[neuron])
        @constraint(jump_model, [neuron = neurons(layer)], s[layer, neuron] <= ub_s[neuron])
    end

    return jump_model, bounds_x, bounds_s
end

function forward_pass!(jump_model::JuMP.Model, input::Vector{Float32})
    @assert length(input) == length(jump_model[:x][0, :]) "Incorrect input length."

    [fix(jump_model[:x][0, i], input[i], force=true) for i in eachindex(input)]
    optimize!(jump_model)

    (last_layer, outputs) = maximum(keys(jump_model[:x].data))
    result = value.(jump_model[:x][last_layer, :])
    return [result[i] for i in 1:outputs]
end