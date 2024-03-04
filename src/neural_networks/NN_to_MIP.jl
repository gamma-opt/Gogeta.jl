"""
    function NN_formulate!(jump_model::JuMP.Model, NN_model::Flux.Chain, U_in, L_in; U_bounds=nothing, L_bounds=nothing, U_out=nothing, L_out=nothing, bound_tightening="fast", compress=false, parallel=false, silent=true)

Creates a mixed-integer optimization problem from a `Flux.Chain` model.

The parameters are used to specify what kind of bound tightening and compression will be used.

# Arguments
- `jump_model`: The constraints and variables will be saved to this optimization model.
- `NN_model`: Neural network model to be formulated.
- `U_in`: Upper bounds for the input variables.
- `L_in`: Lower bounds for the input variables.

# Optional arguments
- `bound_tightening`: Mode selection: "fast", "standard", "output" or "precomputed"
- `compress`: Should the model be simultaneously compressed?
- `parallel`: Runs bound tightening in parallel. `set_solver!`-function must be defined in the global scope, see documentation or examples.
- `U_bounds`: Upper bounds. Needed if bound_tightening="precomputed"
- `L_bounds`: Lower bounds. Needed if bound_tightening="precomputed"
- `U_out`: Upper bounds for the output variables. Needed if bound_tightening="output".
- `L_out`: Lower bounds for the output variables. Needed if bound_tightening="output".
- `silent`: Controls console ouput.

"""
function NN_formulate!(jump_model::JuMP.Model, NN_model::Flux.Chain, U_in, L_in; bound_tightening="fast", compress=false, parallel=false, U_bounds=nothing, L_bounds=nothing, U_out=nothing, L_out=nothing, silent=true)

    oldstdout = stdout
    if silent redirect_stdout(devnull) end

    if compress
        println("Starting compression...")
    else
        println("Creating JuMP model...")
    end

    @assert bound_tightening in ("precomputed", "fast", "standard", "output") "Accepted bound tightening modes are: precomputed, fast, standard, output."

    if bound_tightening == "precomputed" @assert !isnothing(U_bounds) && !isnothing(L_bounds) "Precomputed bounds must be provided." end

    K = length(NN_model) # number of layers (input layer not included)
    W = deepcopy([Flux.params(NN_model)[2*k-1] for k in 1:K])
    b = deepcopy([Flux.params(NN_model)[2*k] for k in 1:K])

    @assert all([NN_model[i].σ == relu for i in 1:K-1]) "Neural network must use the relu activation function."
    @assert NN_model[K].σ == identity "Neural network must use the identity function for the output layer."
    
    removed_neurons = Vector{Vector}(undef, K)
    [removed_neurons[layer] = Vector{Int}() for layer in 1:K]

    input_length = Int((length(W[1]) / length(b[1])))
    neuron_count = [length(b[k]) for k in eachindex(b)]
    neurons(layer) = layer == 0 ? [i for i in 1:input_length] : [i for i in setdiff(1:neuron_count[layer], removed_neurons[layer])]
    @assert input_length == length(U_in) == length(L_in) "Initial bounds arrays must be the same length as the input layer"
        
    @variable(jump_model, x[layer = 0:K, neurons(layer)])
    @variable(jump_model, s[layer = 1:K-1, neurons(layer)])
    @variable(jump_model, z[layer = 1:K-1, neurons(layer)])
        
    @constraint(jump_model, [j = 1:input_length], x[0, j] <= U_in[j])
    @constraint(jump_model, [j = 1:input_length], x[0, j] >= L_in[j])
    
    if bound_tightening != "precomputed" 
        U_bounds = Vector{Vector}(undef, K)
        L_bounds = Vector{Vector}(undef, K)
    end
    
    # upper bound and lower bound constraints for output bound tightening
    ucons = Vector{Vector{ConstraintRef}}(undef, K)
    lcons = Vector{Vector{ConstraintRef}}(undef, K)

    [ucons[layer] = Vector{ConstraintRef}(undef, neuron_count[layer]) for layer in 1:K]
    [lcons[layer] = Vector{ConstraintRef}(undef, neuron_count[layer]) for layer in 1:K]

    layers_removed = 0 # how many strictly preceding layers have been removed at current loop iteration 

    for layer in 1:K # hidden layers and bounds for output layer

        println("\nLAYER $layer")

        if bound_tightening != "precomputed"

            # compute loose bounds
            if layer - layers_removed == 1
                U_bounds[layer] = [sum(max(W[layer][neuron, previous] * U_in[previous], W[layer][neuron, previous] * L_in[previous]) for previous in neurons(layer-1-layers_removed)) + b[layer][neuron] for neuron in neurons(layer)]
                L_bounds[layer] = [sum(min(W[layer][neuron, previous] * U_in[previous], W[layer][neuron, previous] * L_in[previous]) for previous in neurons(layer-1-layers_removed)) + b[layer][neuron] for neuron in neurons(layer)]
            else
                U_bounds[layer] = [sum(max(W[layer][neuron, previous] * max(0, U_bounds[layer-1-layers_removed][previous]), W[layer][neuron, previous] * max(0, L_bounds[layer-1-layers_removed][previous])) for previous in neurons(layer-1-layers_removed)) + b[layer][neuron] for neuron in neurons(layer)]
                L_bounds[layer] = [sum(min(W[layer][neuron, previous] * max(0, U_bounds[layer-1-layers_removed][previous]), W[layer][neuron, previous] * max(0, L_bounds[layer-1-layers_removed][previous])) for previous in neurons(layer-1-layers_removed)) + b[layer][neuron] for neuron in neurons(layer)]
            end

            # compute tighter bounds
            if bound_tightening == "standard"
                bounds = if parallel == true # multiprocessing enabled
                    pmap(neuron -> calculate_bounds(copy_model(jump_model), layer, neuron, W, b, neurons; layers_removed), neurons(layer))
                else
                    map(neuron -> calculate_bounds(jump_model, layer, neuron, W, b, neurons; layers_removed), neurons(layer))
                end
                # only change if bound is improved
                U_bounds[layer] = min.(U_bounds[layer], [bound[1] for bound in bounds])
                L_bounds[layer] = max.(L_bounds[layer], [bound[2] for bound in bounds])
            end
        end

        # output bounds calculated but no more constraints added
        if layer == K
            break
        end

        if compress 
            layers_removed = prune!(W, b, removed_neurons, layers_removed, neuron_count, layer, U_bounds, L_bounds) 
        end

        for neuron in neurons(layer)
            @constraint(jump_model, x[layer, neuron] >= 0)
            @constraint(jump_model, s[layer, neuron] >= 0)
            set_binary(z[layer, neuron])

            ucons[layer][neuron] = @constraint(jump_model, x[layer, neuron] <= max(0, U_bounds[layer][neuron]) * (1 - z[layer, neuron]))
            lcons[layer][neuron] = @constraint(jump_model, s[layer, neuron] <= max(0, -L_bounds[layer][neuron]) * z[layer, neuron])

            @constraint(jump_model, x[layer, neuron] - s[layer, neuron] == b[layer][neuron] + sum(W[layer][neuron, i] * x[layer-1-layers_removed, i] for i in neurons(layer-1-layers_removed)))
        end

        if length(neurons(layer)) > 0
            layers_removed = 0
        end 

    end

    # output layer
    @constraint(jump_model, [neuron in neurons(K)], x[K, neuron] == b[K][neuron] + sum(W[K][neuron, i] * x[K-1-layers_removed, i] for i in neurons(K-1-layers_removed)))

    # using output bounds in bound tightening
    if bound_tightening == "output"
        @assert length(L_out) == length(U_out) == neuron_count[K] "Incorrect length of output bounds array."

        println("Starting bound tightening based on output bounds as well as input bounds.")

        @constraint(jump_model, [neuron in 1:neuron_count[K]], x[K, neuron] >= L_out[neuron])
        @constraint(jump_model, [neuron in 1:neuron_count[K]], x[K, neuron] <= U_out[neuron])

        for layer in 1:K-1

            println("\nLAYER $layer")

            bounds = if parallel == true # multiprocessing enabled
                pmap(neuron -> calculate_bounds(copy_model(jump_model), layer, neuron, W, b, neurons; layers_removed), neurons(layer))
            else
                map(neuron -> calculate_bounds(jump_model, layer, neuron, W, b, neurons; layers_removed), neurons(layer))
            end

            # only change if bound is improved
            U_bounds[layer] = min.(U_bounds[layer], [bound[1] for bound in bounds])
            L_bounds[layer] = max.(L_bounds[layer], [bound[2] for bound in bounds])

            for neuron in neuron_count[layer]

                delete(jump_model, ucons[layer][neuron])
                delete(jump_model, lcons[layer][neuron])

                @constraint(jump_model, x[layer, neuron] <= max(0, U_bounds[layer][neuron]) * (1 - z[layer, neuron]))
                @constraint(jump_model, s[layer, neuron] <= max(0, -L_bounds[layer][neuron]) * z[layer, neuron])
            end

        end

        U_bounds[K] = U_out
        L_bounds[K] = L_out
    end

    redirect_stdout(oldstdout)
    
    if compress
        new_model = build_model!(W, b, K, neurons)

        if bound_tightening != "precomputed"

            U_compressed = [U_bounds[layer][neurons(layer)] for layer in 1:K]
            filter!(neurons -> length(neurons) != 0, U_compressed)

            L_compressed = [L_bounds[layer][neurons(layer)] for layer in 1:K]
            filter!(neurons -> length(neurons) != 0, L_compressed)

            empty!(jump_model)
            NN_formulate!(jump_model, new_model, U_in, L_in; U_bounds=U_compressed, L_bounds=L_compressed, bound_tightening="precomputed", silent)

            return new_model, removed_neurons, U_compressed, L_compressed
        else
            return new_model, removed_neurons
        end
    else
        if bound_tightening != "precomputed"
            return U_bounds, L_bounds
        end
    end

end

"""
    function forward_pass!(jump_model::JuMP.Model, input)

Calculates the output of a JuMP model representing a neural network.
"""
function forward_pass!(jump_model::JuMP.Model, input)
    
    @assert length(input) == length(jump_model[:x][0, :]) "Incorrect input length."

    try
        [fix(jump_model[:x][0, i], input[i], force=true) for i in eachindex(input)]
        optimize!(jump_model)
        (last_layer, outputs) = maximum(keys(jump_model[:x].data))
        result = value.(jump_model[:x][last_layer, :])
        return [result[i] for i in 1:outputs]
    catch e
        @warn "Input or ouput outside of bounds or incorrectly constructed model."
        return [NaN]
    end

end