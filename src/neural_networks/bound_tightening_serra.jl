using BSON
using Flux
using JuMP
using Gurobi

function NN_to_MIP(NN_model::Flux.Chain, init_ub::Vector{Float64}, init_lb::Vector{Float64}; tighten_bounds=false)

    K = length(NN_model) # number of layers (input layer not included)
    W = [Flux.params(NN_model)[2*k-1] for k in 1:K]
    b = [Flux.params(NN_model)[2*k] for k in 1:K]
    
    input_length = Int((length(W[1]) / length(b[1])))
    neuron_count = [length(b[k]) for k in eachindex(b)]
    neurons(layer) = layer == 0 ? [i for i in 1:input_length] : [i for i in 1:neuron_count[layer]]
    
    @assert input_length == length(init_ub) == length(init_lb) "Initial bounds arrays must be the same length as the input layer"
    
    # build model up to second layer
    jump_model = Model()
    set_optimizer(jump_model, () -> Gurobi.Optimizer(ENV))
    set_silent(jump_model)
    set_attribute(jump_model, "TimeLimit", 0.25)
    
    @variable(jump_model, x[layer = 0:K, neurons(layer)])
    @variable(jump_model, s[layer = 0:K, neurons(layer)])
    @variable(jump_model, z[layer = 0:K, neurons(layer)])
    
    @constraint(jump_model, [j = 1:input_length], x[0, j] <= init_ub[j])
    @constraint(jump_model, [j = 1:input_length], x[0, j] >= init_lb[j])
    
    bounds_x = Vector{Vector}(undef, K)
    bounds_s = Vector{Vector}(undef, K)
    
    for layer in 1:K # hidden layers to output layer - second layer and up
    
        ub_x = fill(1000.0, length(neurons(layer)))
        ub_s = fill(1000.0, length(neurons(layer)))

        if tighten_bounds
            for neuron in 1:neuron_count[layer]
                ub_x[neuron], ub_s[neuron] = calculate_bounds(jump_model, layer, neuron, W, b, neurons)
            end
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

    return jump_model, bounds_x, bounds_s
end

function calculate_bounds(model::JuMP.Model, layer, neuron, W, b, neurons)

    @objective(model, Max, b[layer][neuron] + sum(W[layer][neuron, i] * model[:x][layer-1, i] for i in neurons(layer-1)))
    optimize!(model)
    @assert primal_status(model) == MOI.FEASIBLE_POINT "No solution found in time limit."
    
    ub_x = max(objective_bound(model), 0.0)

    set_objective_sense(model, MIN_SENSE)
    optimize!(model)
    @assert primal_status(model) == MOI.FEASIBLE_POINT "No solution found in time limit."
    ub_s = max(-objective_bound(model), 0.0)

    return ub_x, ub_s
end

function forward_pass!(jump_model::JuMP.Model, input::Vector{Float32})
    @assert length(input) == length(jump_model[:x][0, :]) "Incorrect input length."

    [fix(jump_model[:x][0, i], input[i], force=true) for i in eachindex(input)]
    optimize!(jump_model)

    (last_layer, outputs) = maximum(keys(jump_model[:x].data))
    result = value.(jump_model[:x][last_layer, :])
    return [result[i] for i in 1:outputs]
end