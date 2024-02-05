using Flux
using JuMP
using LinearAlgebra: rank, dot

function compress(model::Flux.Chain, init_ub::Vector{Float64}, init_lb::Vector{Float64}, params; big_M=1000.0)

    K = length(model)

    @assert all([model[i].σ == relu for i in 1:K-1]) "Neural network must use the relu activation function."
    @assert model[K].σ == identity "Neural network must use the identity function for the output layer."

    W = deepcopy([Flux.params(model)[2*k-1] for k in 1:K]) # W[i] = weight matrix for i:th layer
    b = deepcopy([Flux.params(model)[2*k] for k in 1:K])

    removed_neurons = Vector{Vector}(undef, K)
    [removed_neurons[layer] = Vector{Int}() for layer in 1:K]

    input_length = Int((length(W[1]) / length(b[1])))
    neuron_count = [length(b[k]) for k in eachindex(b)]
    neurons(layer) = layer == 0 ? [i for i in 1:input_length] : [i for i in setdiff(1:length(b[layer]), removed_neurons[layer])]

    @assert input_length == length(init_ub) == length(init_lb) "Initial bounds arrays must be the same length as the input layer"

    # build JuMP model
    jump_model = direct_model(Gurobi.Optimizer())
    params.silent && set_silent(jump_model)
    params.threads != 0 && set_attribute(jump_model, "Threads", params.threads)
    params.relax && relax_integrality(jump_model)
    params.time_limit != 0 && set_attribute(jump_model, "TimeLimit", params.time_limit)
    
    @variable(jump_model, x[layer = 0:K, neurons(layer)])
    @variable(jump_model, s[layer = 0:K-1, neurons(layer)])
    @variable(jump_model, z[layer = 0:K-1, neurons(layer)])
    
    @constraint(jump_model, [j = 1:input_length], x[0, j] <= init_ub[j])
    @constraint(jump_model, [j = 1:input_length], x[0, j] >= init_lb[j])

    bounds_U = Vector{Vector}(undef, K)
    bounds_L = Vector{Vector}(undef, K)

    for layer in 1:K-1 # hidden layers

        println("\nLAYER: $layer")

        bounds_U[layer] = fill(big_M, length(neurons(layer)))
        bounds_L[layer] = fill(big_M, length(neurons(layer)))

        bounds = map(neuron -> calculate_bounds_fast(jump_model, layer, neuron, W, b, neurons), neurons(layer))
        bounds_U[layer], bounds_L[layer] = [bound[1] > big_M ? big_M : bound[1] for bound in bounds], [bound[2] > big_M ? big_M : bound[2] for bound in bounds]
        println()

        stable_units = Set{Int}() # indices of stable neurons
        unstable_units = false
        
        for neuron in neurons(layer)

            if bounds_U[layer][neuron] <= 0 || iszero(W[layer][neuron, :]) # stabily inactive

                println("Neuron $neuron is stabily inactive")
                
                if neuron < neuron_count[layer] || length(stable_units) > 0 || unstable_units == true
                    
                    if iszero(W[layer][neuron, :]) && b[layer][neuron] > 0
                        
                        for neuron_next in neurons(layer+1)
                            b[layer+1][neuron_next] += W[layer+1][neuron_next, neuron] * b[layer][neuron]
                        end

                    end

                    push!(removed_neurons[layer], neuron)
                end

            elseif bounds_L[layer][neuron] >= 0 # stabily active

                println("Neuron $neuron is stabily active")
                
                if rank(W[layer][collect(union(stable_units, neuron)), :]) > length(stable_units)
                    push!(stable_units, neuron)
                else
                    S = sort!(collect(stable_units))

                    alpha = transpose(W[layer][S, :]) \ W[layer][neuron, :]

                    @assert transpose(W[layer][S, :]) * alpha ≈ W[layer][neuron, :] "Alpha calculation not working."

                    for neuron_next in neurons(layer+1)
                        W[layer+1][neuron_next, S] .+= sum(W[layer+1][neuron_next, neuron] * alpha)
                        b[layer+1][neuron_next] += W[layer+1][neuron_next, neuron] * (b[layer][neuron] + dot(b[layer][S], alpha))
                    end

                    push!(removed_neurons[layer], neuron)
                end
            else
                unstable_units = true
            end

        end

        println()

        if unstable_units == false # all units in the layer are stable
            println("Fully stable layer")

            if length(stable_units) > 0
                # TODO implement folding code
            else
                output = model((init_ub + init_lb) ./ 2)
                println("WHOLE NETWORK IS CONSTANT WITH OUTPUT: $output")
                return
            end
        end

        println("Removed $(length(removed_neurons[layer])) neurons")

        for neuron in neurons(layer)
            @constraint(jump_model, x[layer, neuron] >= 0)
            @constraint(jump_model, s[layer, neuron] >= 0)
            set_binary(z[layer, neuron])

            @constraint(jump_model, x[layer, neuron] <= bounds_U[layer][neuron] * (1 - z[layer, neuron]))
            @constraint(jump_model, s[layer, neuron] <= -bounds_L[layer][neuron] * z[layer, neuron])

            @constraint(jump_model, x[layer, neuron] - s[layer, neuron] == b[layer][neuron] + sum(W[layer][neuron, i] * x[layer-1, i] for i in neurons(layer-1)))
        end

    end

    # output layer
    @constraint(jump_model, [neuron in neurons(K)], x[K, neuron] == b[K][neuron] + sum(W[K][neuron, i] * x[K-1, i] for i in neurons(K-1)))

    # build compressed model
    new_layers = [];
    for layer in 1:K
        if layer != K
            W[layer] = W[layer][setdiff(1:size(W[layer])[1], removed_neurons[layer]), setdiff(1:size(W[layer])[2], layer == 1 ? [] : removed_neurons[layer-1])]
            b[layer] = b[layer][setdiff(1:length(b[layer]), removed_neurons[layer])]

            push!(new_layers, Dense(W[layer], b[layer], relu))
        else
            push!(new_layers, Dense(W[layer], b[layer]))
        end
    end

    new_model = Flux.Chain(new_layers...)

    return jump_model, removed_neurons, new_model, bounds_U, bounds_L

end