"""
    function prune!(W, b, removed_neurons, layers_removed, neuron_count, layer, bounds_U, bounds_L)

Removes stabily active or inactive neurons in a network by updating the weights and the biases and the removed neurons list accordingly.
"""
function prune!(W, b, removed_neurons, layers_removed, neuron_count, layer, bounds_U, bounds_L)

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
            else  # neuron is linearly dependent

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
            removed_neurons[layer] = 1:neuron_count[layer]
        else
            output = model((init_ub + init_lb) ./ 2)
            error("WHOLE NETWORK IS CONSTANT WITH OUTPUT: $output")
        end
    end

    println("Removed $(length(removed_neurons[layer]))/$(neuron_count[layer]) neurons")
    return layers_removed
end

"""
    function build_model!(W, b, K, neurons)

Builds a new `Flux.Chain` model from the given weights and biases.
"""
function build_model!(W, b, K, neurons)

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

    return Flux.Chain(new_layers...)

end