"""
    function NN_incorporate!(
        jump_original::JuMP.Model,
        param_source,
        output_var,
        input_vars...;
        U_in,
        L_in,
        compress=false,
        bound_tightening="fast",
        parallel=false
    )

Formulates the neural network (NN) as a MIP into a JuMP model. The model parameters must be contained in a JSON file located at the given filepath OR in a Flux.Chain model.

JSON file structure:
- Parameter sets must be named based on the layer type  - "FCx" where 'x' indicates the layer number.
- 'FC' indicates a fully connected layer. These parameter sets contain weights and biases.
- For a clear example on how to export these parameters from Tensorflow, see examples/-folder of the package repository

This function modifies the JuMP model given as input. The input and output variable references that are given as input to this function will be linked to the appropriate variables in the NN formulation. No additional names are added to the JuMP model - all variables are added as anonymous. 

Different bound tightening modes and compression can be used. To use bound tightening modes other than "fast", `set_solver!`-function must be defined in the global scope which defines the solver and other necessary for the JuMP model used in bound tightening.

# Arguments

- `jump_original`: JuMP model into which the neural network MIP should be incorporated
- `param_source`: relative path to the JSON file containing the model parameters OR a Flux.Chain model
- `output_var`: reference to the variable that should be linked to the NN output
- `input_vars`: references to the variables that will be used as the NN inputs

# Keyword Arguments

- `U_in`: vector of upper bounds for the input variables
- `L_in`: vector of lower bounds for the input variables
- `compress`: reduce NN size by removing stable and linearly dependent neurons
- `bound_tightening`: which bound tightening mode to use: "fast", "standard", "standard_linear", "output"
- `parallel`: use multiprocessing for bound tightening

"""
function NN_incorporate!(
    jump_original::JuMP.Model,
    param_source,
    output_var,
    input_vars...;
    U_in,
    L_in,
    compress=false,
    bound_tightening="fast",
    parallel=false
)

    W, b = if typeof(param_source) == String
        get_JSON_params(param_source)
    elseif typeof(param_source) == Flux.Chain
        get_Flux_params(param_source)
    else
        @error "Model must be either a Flux.Chain or a filepath containing a JSON file."
    end

    K = length(b)
    removed_neurons = Vector{Vector}(undef, K)
    [removed_neurons[layer] = Vector{Int}() for layer in 1:K]

    input_length = Int((length(W[1]) / length(b[1])))
    neuron_count = [length(b[k]) for k in eachindex(b)]
    neurons(layer) = layer == 0 ? [i for i in 1:input_length] : [i for i in setdiff(1:neuron_count[layer], removed_neurons[layer])]
    @assert input_length == length(U_in) == length(L_in) "Initial bounds arrays must be the same length as the input layer"
        
    U_bounds = Vector{Vector}(undef, K)
    L_bounds = Vector{Vector}(undef, K)

    jump_model = Model()
    if bound_tightening != "fast"
        Main.set_solver!(jump_model)
    
        @variable(jump_model, x[layer = 0:K, neurons(layer)])
        @variable(jump_model, s[layer = 1:K-1, neurons(layer)])
        @variable(jump_model, z[layer = 1:K-1, neurons(layer)], Bin)
            
        @constraint(jump_model, [j = 1:input_length], x[0, j] <= U_in[j])
        @constraint(jump_model, [j = 1:input_length], x[0, j] >= L_in[j])

        if bound_tightening == "standard_linear"
            relax_integrality(jump_model)
        end
    
        # upper bound and lower bound constraints for output bound tightening
        ucons = Vector{Vector{ConstraintRef}}(undef, K)
        lcons = Vector{Vector{ConstraintRef}}(undef, K)
    
        [ucons[layer] = Vector{ConstraintRef}(undef, neuron_count[layer]) for layer in 1:K]
        [lcons[layer] = Vector{ConstraintRef}(undef, neuron_count[layer]) for layer in 1:K]
    end

    layers_removed = 0 # how many strictly preceding layers have been removed at current loop iteration 

    for layer in 1:K # hidden layers and bounds for output layer

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
                pmap(neuron -> calculate_bounds(copy_model(jump_model), layer, neuron, W[layer], b[layer], neurons; layers_removed), neurons(layer))
            else
                map(neuron -> calculate_bounds(jump_model, layer, neuron, W[layer], b[layer], neurons; layers_removed), neurons(layer))
            end
            # only change if bound is improved
            U_bounds[layer] = min.(U_bounds[layer], [bound[1] for bound in bounds])
            L_bounds[layer] = max.(L_bounds[layer], [bound[2] for bound in bounds])
        end

        # output bounds calculated but no more constraints added
        if layer == K
            break
        end

        if compress 
            layers_removed = prune!(W, b, removed_neurons, layers_removed, neuron_count, layer, U_bounds, L_bounds) 
        end

        if bound_tightening != "fast"
            for neuron in neurons(layer)
                @constraint(jump_model, x[layer, neuron] >= 0)
                @constraint(jump_model, s[layer, neuron] >= 0)

                ucons[layer][neuron] = @constraint(jump_model, x[layer, neuron] <= max(0, U_bounds[layer][neuron]) * z[layer, neuron])
                lcons[layer][neuron] = @constraint(jump_model, s[layer, neuron] <= max(0, -L_bounds[layer][neuron]) * (1-z[layer, neuron]))

                @constraint(jump_model, x[layer, neuron] - s[layer, neuron] == b[layer][neuron] + sum(W[layer][neuron, i] * x[layer-1-layers_removed, i] for i in neurons(layer-1-layers_removed)))
            end
        end

        if length(neurons(layer)) > 0
            layers_removed = 0
        end 

    end
    
    # using output bounds in bound tightening
    if bound_tightening == "output"  
        # output layer
        @constraint(jump_model, [neuron in neurons(K)], x[K, neuron] == b[K][neuron] + sum(W[K][neuron, i] * x[K-1-layers_removed, i] for i in neurons(K-1-layers_removed)))

        @assert length(L_out) == length(U_out) == neuron_count[K] "Incorrect length of output bounds array."

        @constraint(jump_model, [neuron in 1:neuron_count[K]], x[K, neuron] >= L_out[neuron])
        @constraint(jump_model, [neuron in 1:neuron_count[K]], x[K, neuron] <= U_out[neuron])

        for layer in 1:K-1

            bounds = if parallel == true # multiprocessing enabled
                pmap(neuron -> calculate_bounds(copy_model(jump_model), layer, neuron, W[layer], b[layer], neurons; layers_removed), neurons(layer))
            else
                map(neuron -> calculate_bounds(jump_model, layer, neuron, W[layer], b[layer], neurons; layers_removed), neurons(layer))
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

    # incorporate NN formulation into original JuMP model using bounds that have just been computed
    if compress
        new_model = build_model!(W, b, K, neurons)
        W, b = get_Flux_params(new_model)

        U_compressed = [U_bounds[layer][neurons(layer)] for layer in 1:K]
        filter!(neurons -> length(neurons) != 0, U_compressed)

        L_compressed = [L_bounds[layer][neurons(layer)] for layer in 1:K]
        filter!(neurons -> length(neurons) != 0, L_compressed)

        U_bounds = U_compressed
        L_bounds = L_compressed
    end

    anon_NN_from_bounds!(jump_original, W, b, output_var, input_vars...; U_in, L_in, U_bounds, L_bounds)

end

"""
    function anon_NN_from_bounds!(jump::JuMP.Model, W, b, output_var, input_vars...; U_in, L_in, U_bounds, L_bounds)

Helper function for anonymously formulating the NN into a JuMP model given weights, biases and the variable references to be linked.
"""
function anon_NN_from_bounds!(jump::JuMP.Model, W, b, output_var, input_vars...; U_in, L_in, U_bounds, L_bounds)

    layers = 1:length(b)
    n_neurons = Dict{Int, Int}()
    n_neurons[0] = length(U_in)
    [n_neurons[layer] = length(b[layer]) for layer in layers]

    x = @variable(jump, [layer=union(0, layers), 1:n_neurons[layer]])
    s = @variable(jump, [layer=layers[1:end-1], 1:n_neurons[layer]])
    z = @variable(jump, [layer=layers[1:end-1], 1:n_neurons[layer]])
    @constraint(jump, [layer=layers[1:end-1], neuron=1:n_neurons[layer]], x[layer, neuron] >= 0)
    @constraint(jump, [layer=layers[1:end-1], neuron=1:n_neurons[layer]], s[layer, neuron] >= 0)

    input = [var for var in input_vars]
    @constraint(jump, [neuron=1:n_neurons[0]], x[0, neuron] == input[neuron])
    @constraint(jump, [neuron=1:n_neurons[0]], L_in[neuron] <= x[0, neuron] <= U_in[neuron])

    for layer in layers # 1...last layer

        for neuron in 1:n_neurons[layer]
            
            FC_sum = @expression(jump, sum(W[layer][neuron, neuron_last] * x[layer-1, neuron_last] for neuron_last in 1:n_neurons[layer-1]))
            bias = b[layer][neuron]
            
            if layer == maximum(layers)
                @constraint(jump, x[layer, neuron] == FC_sum + bias)
                break
            end

            set_binary(z[layer, neuron])
            @constraint(jump, x[layer, neuron] <= max(0, U_bounds[layer][neuron]) * z[layer, neuron])
            @constraint(jump, s[layer, neuron] <= max(0, -L_bounds[layer][neuron]) * (1-z[layer, neuron]))

            @constraint(jump, x[layer, neuron] - s[layer, neuron] == FC_sum + bias)
        end
    end

    # reference to output variable
    @constraint(jump, output_var == x[maximum(layers), 1])

end

"""
    function forward_pass_NN!(jump, input, output_var, input_vars)

Calculates the forward pass through the NN (the output that is produced with the given input).
This function can be used when the NN formulation is incorporated into a larger optimization problem.

# Arguments

- `jump`: JuMP model with the NN formulation
- `input`: value to be forward passed
- `output_var`: reference to the NN output variable
- `input_vars`: references to the NN input variables

"""
function forward_pass_NN!(jump, input, output_var, input_vars)

    input_var = [var for var in input_vars]
    [fix(input_var[neuron], input[neuron]) for neuron in eachindex(input)]
    
    try
        optimize!(jump)
        return value(output_var)
    catch e
        println("Error with forward pass: $e")
        @warn "Input or output outside of bounds or incorrectly constructed model."
        return NaN
    end
end

"""
    function get_Flux_params(NN_model::Flux.Chain)

Helper function for getting the necessary parameters out of a Flux.Chain model.
Returns the weights and the biases as vectors.
"""
function get_Flux_params(NN_model::Flux.Chain)

    K = length(NN_model) # number of layers (input layer not included)
    W = deepcopy([Flux.params(NN_model)[2*k-1] for k in 1:K])
    b = deepcopy([Flux.params(NN_model)[2*k] for k in 1:K])

    @assert all([NN_model[i].σ == relu for i in 1:K-1]) "Neural network must use the relu activation function."
    @assert NN_model[K].σ == identity "Neural network must use the identity function for the output layer."

    return W, b
end

"""
    function get_JSON_params(filepath::String)

Helper function for getting the necessary parameters out of a JSON file.
Returns the weights and the biases as vectors.
"""
function get_JSON_params(filepath::String)

    params = JSON.parsefile(filepath)
    # Weights: W[layer name][1][column num][row]
    # Biases: W[layer name][2][bias index]
    # names FC-1,2,...,k (fully connected)
    #       (k is output layer index)

    layers = parse.(Int, last.(filter(key -> occursin("FC", key), keys(params)))) # FC indices
    sort!(layers)

    W = [params["FC"*string(layer)][1] for layer in layers]
    b = [params["FC"*string(layer)][2] for layer in layers]

    W = [reduce(hcat, W[layer]) for layer in layers]

    return W, b

end