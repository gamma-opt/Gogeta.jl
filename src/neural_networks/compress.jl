using Flux
using JuMP
using LinearAlgebra: rank, dot

# function compress(model::Flux.Chain, init_ub::Vector{Float64}, init_lb::Vector{Float64}, params; big_M=1000.0)

#     K = length(model)

#     @assert all([model[i].σ == relu for i in 1:K-1]) "Neural network must use the relu activation function."
#     @assert model[K].σ == identity "Neural network must use the identity function for the output layer."

#     W = deepcopy([Flux.params(model)[2*k-1] for k in 1:K]) # W[i] = weight matrix for i:th layer
#     b = deepcopy([Flux.params(model)[2*k] for k in 1:K])

#     removed_neurons = Vector{Vector}(undef, K)
#     [removed_neurons[layer] = Vector{Int}() for layer in 1:K]

#     input_length = Int((length(W[1]) / length(b[1])))
#     neuron_count = [length(b[k]) for k in eachindex(b)]
#     neurons(layer) = layer == 0 ? [i for i in 1:input_length] : [i for i in setdiff(1:length(b[layer]), removed_neurons[layer])]

#     @assert input_length == length(init_ub) == length(init_lb) "Initial bounds arrays must be the same length as the input layer"

#     # build JuMP model
#     jump_model = direct_model(Gurobi.Optimizer())
#     params.silent && set_silent(jump_model)
#     params.threads != 0 && set_attribute(jump_model, "Threads", params.threads)
#     params.relax && relax_integrality(jump_model)
#     params.time_limit != 0 && set_attribute(jump_model, "TimeLimit", params.time_limit)
    
#     @variable(jump_model, x[layer = 0:K, neurons(layer)])
#     @variable(jump_model, s[layer = 1:K-1, neurons(layer)])
#     @variable(jump_model, z[layer = 1:K-1, neurons(layer)])
    
#     @constraint(jump_model, [j = 1:input_length], x[0, j] <= init_ub[j])
#     @constraint(jump_model, [j = 1:input_length], x[0, j] >= init_lb[j])

#     bounds_U = Vector{Vector}(undef, K)
#     bounds_L = Vector{Vector}(undef, K)

#     layers_removed = 0

#     for layer in 1:K-1 # hidden layers

#         println("\nLAYER: $layer")

#         bounds_U[layer] = fill(big_M, length(neurons(layer)))
#         bounds_L[layer] = fill(big_M, length(neurons(layer)))

#         bounds = map(neuron -> calculate_bounds_fast(jump_model, layer, neuron, W, b, neurons, layers_removed), neurons(layer))
#         bounds_U[layer], bounds_L[layer] = [bound[1] > big_M ? big_M : bound[1] for bound in bounds], [bound[2] > big_M ? big_M : bound[2] for bound in bounds]
#         println()

#         stable_units = Set{Int}() # indices of stable neurons
#         unstable_units = false
        
#         for neuron in neurons(layer)

#             if bounds_U[layer][neuron] <= 0 || iszero(W[layer][neuron, :]) # stabily inactive

#                 #println("Neuron $neuron is stabily inactive")
                
#                 if neuron < last(neurons(layer)) || length(stable_units) > 0 || unstable_units == true
                    
#                     if iszero(W[layer][neuron, :]) && b[layer][neuron] > 0
                        
#                         for neuron_next in neurons(layer+1)
#                             b[layer+1][neuron_next] += W[layer+1][neuron_next, neuron] * b[layer][neuron]
#                         end

#                     end

#                     push!(removed_neurons[layer], neuron)
#                 end

#             elseif bounds_L[layer][neuron] >= 0 # stabily active

#                 #println("Neuron $neuron is stabily active")
                
#                 if rank(W[layer][collect(union(stable_units, neuron)), :]) > length(stable_units)
#                     push!(stable_units, neuron)
#                 else
#                     S = sort!(collect(stable_units))

#                     alpha = transpose(W[layer][S, :]) \ W[layer][neuron, :]

#                     @assert transpose(W[layer][S, :]) * alpha ≈ W[layer][neuron, :] "Alpha calculation not working."

#                     for neuron_next in neurons(layer+1)
#                         W[layer+1][neuron_next, S] .+= sum(W[layer+1][neuron_next, neuron] * alpha)
#                         b[layer+1][neuron_next] += W[layer+1][neuron_next, neuron] * (b[layer][neuron] + dot(b[layer][S], alpha))
#                     end

#                     push!(removed_neurons[layer], neuron)
#                 end
#             else
#                 unstable_units = true
#             end

#         end

#         println()

#         if unstable_units == false # all units in the layer are stable
#             println("Fully stable layer")

#             if length(stable_units) > 0

#                 W_bar = Matrix{eltype(W[1][1])}(undef, neuron_count[layer+1], neuron_count[layer-1])
#                 b_bar = Vector{eltype(b[1][1])}(undef, neuron_count[layer+1])

#                 for neuron_next in neurons(layer+1)

#                     b_bar[neuron_next] = b[layer+1][neuron_next] + sum([W[layer+1][neuron_next, k] * b[layer][k] for k in collect(stable_units)])

#                     for neuron_previous in neurons(layer-1)
#                         W_bar[neuron_next, neuron_previous] = sum([W[layer+1][neuron_next, k] * W[layer][k, neuron_previous] for k in collect(stable_units)])
#                     end
#                 end

#                 W[layer+1] = W_bar
#                 b[layer+1] = b_bar

#                 layers_removed += 1
#                 push!(removed_neurons[layer], neurons(layer)...)
#             else
#                 output = model((init_ub + init_lb) ./ 2)
#                 println("WHOLE NETWORK IS CONSTANT WITH OUTPUT: $output")
#                 return
#             end
#         end

#         println("Removed $(length(removed_neurons[layer]))/$(neuron_count[layer]) neurons")

#         for neuron in neurons(layer)
#             @constraint(jump_model, x[layer, neuron] >= 0)
#             @constraint(jump_model, s[layer, neuron] >= 0)
#             set_binary(z[layer, neuron])

#             @constraint(jump_model, x[layer, neuron] <= bounds_U[layer][neuron] * (1 - z[layer, neuron]))
#             @constraint(jump_model, s[layer, neuron] <= -bounds_L[layer][neuron] * z[layer, neuron])

#             @constraint(jump_model, x[layer, neuron] - s[layer, neuron] == b[layer][neuron] + sum(W[layer][neuron, i] * x[layer-1-layers_removed, i] for i in neurons(layer-1-layers_removed)))
#         end

#         if length(neurons(layer)) > 0
#             layers_removed = 0
#         end

#     end

#     # output layer
#     @constraint(jump_model, [neuron in neurons(K)], x[K, neuron] == b[K][neuron] + sum(W[K][neuron, i] * x[K-1-layers_removed, i] for i in neurons(K-1-layers_removed)))

#     println("Weights: $([size(W[layer]) for layer in 1:K])")

#     # build compressed model
#     new_layers = [];
#     layers = findall(neurons -> length(neurons) > 0, [neurons(l) for l in 1:K]) # exclude layers with no neurons
#     for (i, layer) in enumerate(layers)

#         if layers[i-1] != layer-1 # previous layer was removed
#         else
#             W[layer] = W[layer][setdiff(1:size(W[layer])[1], removed_neurons[layer]), setdiff(1:size(W[layer])[2], layer == 1 ? [] : removed_neurons[layers[i-1]])]
#             b[layer] = b[layer][setdiff(1:length(b[layer]), removed_neurons[layer])]
#         end

#         if layer != last(layers)
#             push!(new_layers, Dense(W[layer], b[layer], relu))
#         else
#             push!(new_layers, Dense(W[layer], b[layer]))
#         end
#     end

#     new_model = Flux.Chain(new_layers...)

#     return jump_model, removed_neurons, new_model, bounds_U, bounds_L

# end

function compress(model::Flux.Chain, init_ub::Vector{Float64}, init_lb::Vector{Float64}, bounds_U, bounds_L)

    K = length(model)

    @assert all([model[i].σ == relu for i in 1:K-1]) "Neural network must use the relu activation function."
    @assert model[K].σ == identity "Neural network must use the identity function for the output layer."

    W = deepcopy([Flux.params(model)[2*k-1] for k in 1:K]) # W[i] = weight matrix for i:th layer
    b = deepcopy([Flux.params(model)[2*k] for k in 1:K])

    removed_neurons = Vector{Vector}(undef, K)
    [removed_neurons[layer] = Vector{Int}() for layer in 1:K]

    input_length = Int((length(W[1]) / length(b[1])))
    neuron_count = [length(b[k]) for k in eachindex(b)]
    neurons(layer) = layer == 0 ? [i for i in 1:input_length] : [i for i in setdiff(1:neuron_count[layer], removed_neurons[layer])]

    layers_removed = 0 # how many strictly preceding layers have been removed at current loop iteration 

    for layer in 1:K-1 # hidden layers

        println("\nLAYER $layer")

        stable_units = Set{Int}() # indices of stable neurons
        unstable_units = false
        
        for neuron in 1:neuron_count[layer]

            if bounds_U[layer][neuron] <= 0 || iszero(W[layer][neuron, :]) # stabily inactive

                if neuron < neuron_count[layer] || length(stable_units) > 0 || unstable_units == true
                    
                    if iszero(W[layer][neuron, :]) && b[layer][neuron] > 0
                        for neuron_next in 1:neuron_count[layer+1] # adjust biases
                            b[layer+1][neuron_next] += W[layer+1][neuron_next, neuron] * b[layer][neuron]
                        end
                    end

                    push!(removed_neurons[layer], neuron)
                end

            elseif bounds_L[layer][neuron] >= 0 # stabily active
                
                if rank(W[layer][collect(union(stable_units, neuron)), :]) > length(stable_units)
                    push!(stable_units, neuron)
                else

                    S = collect(stable_units)
                    alpha = transpose(W[layer][S, :]) \ W[layer][neuron, :]

                    for neuron_next in 1:neuron_count[layer+1] # adjust weights and biases
                        W[layer+1][neuron_next, S] .+= W[layer+1][neuron_next, neuron] * alpha
                        b[layer+1][neuron_next] += W[layer+1][neuron_next, neuron] * (b[layer][neuron] - dot(b[layer][S], alpha))
                    end

                    push!(removed_neurons[layer], neuron)
                end
            else
                unstable_units = true
            end

        end

        if unstable_units == false # all units in the layer are stable
            println("Fully stable layer")

            if length(stable_units) > 0

                W_bar = Matrix{eltype(W[1][1])}(undef, neuron_count[layer+1], neuron_count[layer-1-layers_removed])
                b_bar = Vector{eltype(b[1][1])}(undef, neuron_count[layer+1])

                S = collect(stable_units)

                for neuron_next in 1:neuron_count[layer+1]

                    b_bar[neuron_next] = b[layer+1][neuron_next] + dot(W[layer+1][neuron_next, S], b[layer][S])

                    for neuron_previous in 1:neuron_count[layer-1-layers_removed]
                        W_bar[neuron_next, neuron_previous] = dot(W[layer+1][neuron_next, S], W[layer][S, neuron_previous])
                    end
                end

                W[layer+1] = W_bar
                b[layer+1] = b_bar

                layers_removed += 1
                push!(removed_neurons[layer], neurons(layer)...)
            else
                output = model((init_ub + init_lb) ./ 2)
                println("WHOLE NETWORK IS CONSTANT WITH OUTPUT: $output")
                return output
            end
        end

        println("Removed $(length(removed_neurons[layer]))/$(neuron_count[layer]) neurons")

        if length(neurons(layer)) > 0
            layers_removed = 0
        end 

    end

    # build compressed model
    new_layers = [];
    layers = findall(neurons -> length(neurons) > 0, [neurons(l) for l in 1:K]) # layers with neurons
    for (i, layer) in enumerate(layers)

        W[layer] = W[layer][neurons(layer), neurons(i == 1 ? 0 : layers[i-1])]
        b[layer] = b[layer][neurons(layer)]

        if layer != last(layers)
            push!(new_layers, Dense(W[layer], b[layer], relu))
        else
            push!(new_layers, Dense(W[layer], b[layer]))
        end
    end

    new_model = Flux.Chain(new_layers...)

    return nothing, removed_neurons, new_model, bounds_U, bounds_L

end