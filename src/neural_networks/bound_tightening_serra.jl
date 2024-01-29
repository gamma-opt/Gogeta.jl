using Flux
using JuMP
using Distributed

function NN_to_MIP(NN_model::Flux.Chain, init_ub::Vector{Float64}, init_lb::Vector{Float64}; tighten_bounds=false, big_M=1000.0)

    K = length(NN_model) # number of layers (input layer not included)
    @assert reduce(&, [NN_model[i].σ == relu for i in 1:K-1]) "Neural network must use the relu activation function."
    @assert NN_model[K].σ == identity "Neural network must use the identity function for the output layer."

    W = [Flux.params(NN_model)[2*k-1] for k in 1:K]
    b = [Flux.params(NN_model)[2*k] for k in 1:K]
    
    input_length = Int((length(W[1]) / length(b[1])))
    neuron_count = [length(b[k]) for k in eachindex(b)]
    neurons(layer) = layer == 0 ? [i for i in 1:input_length] : [i for i in 1:neuron_count[layer]]
    
    @assert input_length == length(init_ub) == length(init_lb) "Initial bounds arrays must be the same length as the input layer"
    
    # build model up to second layer
    jump_model = Model()
    set_solver_params!(jump_model)
    
    @variable(jump_model, x[layer = 0:K, neurons(layer)])
    @variable(jump_model, s[layer = 0:K-1, neurons(layer)])
    @variable(jump_model, z[layer = 0:K-1, neurons(layer)])
    
    @constraint(jump_model, [j = 1:input_length], x[0, j] <= init_ub[j])
    @constraint(jump_model, [j = 1:input_length], x[0, j] >= init_lb[j])
    
    bounds_x = Vector{Vector}(undef, K-1)
    bounds_s = Vector{Vector}(undef, K-1)
    
    for layer in 1:K-1 # hidden layers
    
        ub_x = fill(big_M, length(neurons(layer)))
        ub_s = fill(big_M, length(neurons(layer)))

        println("\nLAYER $layer")

        if tighten_bounds
            bounds = map(neuron -> calculate_bounds(copy_model(jump_model), layer, neuron, W, b, neurons), neurons(layer))
            ub_x, ub_s = [bound[1] for bound in bounds], [bound[2] for bound in bounds]
        end

        for neuron in 1:neuron_count[layer]

            @constraint(jump_model, x[layer, neuron] >= 0)
            @constraint(jump_model, s[layer, neuron] >= 0)
            set_binary(z[layer, neuron])

            @constraint(jump_model, x[layer, neuron] <= ub_x[neuron] * (1 - z[layer, neuron]))
            @constraint(jump_model, s[layer, neuron] <= ub_s[neuron] * z[layer, neuron])

            @constraint(jump_model, x[layer, neuron] - s[layer, neuron] == b[layer][neuron] + sum(W[layer][neuron, i] * x[layer-1, i] for i in neurons(layer-1)))

        end
        
        bounds_x[layer] = ub_x
        bounds_s[layer] = ub_s
    end

    # output layer
    @constraint(jump_model, [neuron in 1:neuron_count[K]], x[K, neuron] == b[K][neuron] + sum(W[K][neuron, i] * x[K-1, i] for i in neurons(K-1)))

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