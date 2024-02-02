using Flux
using JuMP
using LinearAlgebra: rank, dot

function compress(model::Flux.Chain, init_ub::Vector{Float64}, init_lb::Vector{Float64}, params; big_M=1000.0)

    K = length(model)

    @assert all([model[i].σ == relu for i in 1:K-1]) "Neural network must use the relu activation function."
    @assert model[K].σ == identity "Neural network must use the identity function for the output layer."

    W = [Flux.params(model)[2*k-1] for k in 1:K] # W[i] = weight matrix for i:th layer
    b = [Flux.params(model)[2*k] for k in 1:K]

    input_length = Int((length(W[1]) / length(b[1])))
    neuron_count = [length(b[k]) for k in eachindex(b)]
    neurons(layer) = layer == 0 ? [i for i in 1:input_length] : [i for i in 1:length(b[layer])]

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

    removed_neurons = Vector{Vector}(undef, K-1)

    for layer in 1:K-1 # hidden layers

        println("\nLAYER: $layer")

        bounds = map(neuron -> calculate_bounds_fast(jump_model, layer, neuron, W, b, neurons), neurons(layer))

        stable_units = Set{Int}() # indices of stable neurons
        unstable_units = false

        removed_neurons[layer] = Vector{Int}()
        
        for neuron in neurons(layer)

            if bounds[neuron][1] == false || iszero(W[layer][neuron, :]) # constant output
                
                if length(stable_units) > 0 || unstable_units == true || neuron < neuron_count[layer]
                    
                    if iszero(W[layer][neuron, :]) && b[layer][neuron] > 0
                        
                        for neuron_next in neurons(layer+1)
                            b[layer+1][neuron_next] += W[layer+1][neuron_next, neuron] * b[layer][neuron]
                        end

                    end

                    # TODO neurons() function has to be changed as the layers are pruned

                    # W[layer] = W[layer][setdiff(neurons(layer), neuron), :]
                    # b[layer] = b[layer][setdiff(neurons(layer), neuron)]
                    push!(removed_neurons[layer], neuron)
                end

            elseif bounds[neuron][2] == false # stabily active
                
                if rank(W[layer][collect(union(stable_units, neuron))]) > length(stable_units)
                    push!(stable_units, neuron)
                else
                    alpha = W[layer][collect(stable_units), :]' \ W[layer][neuron, :]
                    @assert dot(alpha, W[layer][collect(stable_units), :]') == W[layer][neuron, :] "Alpha calculation not working."

                    for neuron_next in neurons(layer+1)
                        W[layer+1][neuron_next, collect(stable_units)] .+= sum(W[layer+1][neuron_next, neuron] * alpha)
                        b[layer+1][neuron_next] += W[layer+1][neuron_next, neuron] * (b[layer][neuron] + dot(b[layer][collect(stable_units)], alpha))
                    end

                    # W[layer] = W[layer][setdiff(neurons(layer), neuron), :]
                    # b[layer] = b[layer][setdiff(neurons(layer), neuron)]
                    push!(removed_neurons[layer], neuron)
                end
            else
                unstable_units = true
            end

        end

        if unstable_units == false # all units in the layer are stable
            println("Fully stable layer")
            # TODO implement folding code
        end

        for neuron in 1:(neuron_count[layer] - length(removed_neurons[layer]))
            @constraint(jump_model, x[layer, neuron] >= 0)
            @constraint(jump_model, s[layer, neuron] >= 0)
            set_binary(z[layer, neuron])

            @constraint(jump_model, x[layer, neuron] <= big_M * (1 - z[layer, neuron]))
            @constraint(jump_model, s[layer, neuron] <= big_M * z[layer, neuron])

            @constraint(jump_model, x[layer, neuron] - s[layer, neuron] == b[layer][neuron] + sum(W[layer][neuron, i] * x[layer-1, i] for i in neurons(layer-1)))
        end

    end

    @constraint(jump_model, [neuron in 1:neuron_count[K]], x[K, neuron] == b[K][neuron] + sum(W[K][neuron, i] * x[K-1, i] for i in neurons(K-1)))

    return jump_model, removed_neurons

end