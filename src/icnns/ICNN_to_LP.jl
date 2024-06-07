"""
    function ICNN_formulate!(jump::JuMP.Model, filepath::String)

Formulates the input convex neural network (ICNN) as an LP into a JuMP model. The model parameters must be contained in a JSON file located in the given filepath. For more information about ICNNs see Amos et al. (2017).

JSON file structure:
- Parameter sets must be named based on the layer type  - either "FCx" or "SKIPx" where 'x' indicates the layer number.
- 'FC' indicates a fully connected layer. These parameter sets contain weights and biases.
- 'SKIP' indicates a skip connection layer. These only contain weights.
- For a clear example on how to export these parameters from Tensorflow, see examples/-folder of the package repository

This function modifes the JuMP model given as input and outputs variables references to the ICNN input and output variables in the JuMP model. Also modifies the JuMP model objective function to include the ICNN output as a penalty term.

# Arguments

- `jump`: JuMP model where the LP formulation should be saved
- `filepath`: relative path to the JSON file containing the model parameters

"""
function ICNN_formulate!(jump::JuMP.Model, filepath::String)

    W = JSON.parsefile(filepath)
    # Weights: W[layer name][1][column num][row]
    # Biases: W[layer name][2][bias index]
    # names SKIP-2,3,...,k (skip connection)
    #       FC-1,2,...,k (fully connected)
    #       (k is output layer index)

    layers = parse.(Int, last.(filter(key -> occursin("FC", key), keys(W)))) # FC indices
    sort!(layers)

    n_neurons = Dict{Int, Int}() # layer => num. of neurons
    n_neurons[0] = length(W["FC1"][1]) # input length
    for layer in layers
        n_neurons[layer] = length(W["FC"*string(layer)][2])
    end

    @variable(jump, z[layer=union(0, layers), 1:n_neurons[layer]])
    @constraint(jump, [layer=layers[1:end-1], neuron=1:n_neurons[layer]], z[layer, neuron] >= 0)

    for layer in layers
        for neuron in 1:n_neurons[layer]

            FC_sum = @expression(jump, sum([W["FC"*string(layer)][1][neuron_last][neuron] * z[layer-1, neuron_last] for neuron_last in 1:n_neurons[layer-1]]))
            bias = @expression(jump, W["FC"*string(layer)][2][neuron])
            
            if layer == 1
                @constraint(jump, z[layer, neuron] >= FC_sum + bias)
            else
                SKIP_sum = @expression(jump, sum([W["SKIP"*string(layer)][1][input][neuron] * z[0, input] for input in 1:n_neurons[0]]))
                @constraint(jump, z[layer, neuron] >= FC_sum + SKIP_sum + bias)
            end
        end
    end

    # reference to input and output variables
    input_var = jump[:z][0, :]
    output_var = jump[:z][maximum(layers), 1]

    @assert objective_sense(jump) != MAX_SENSE "Objective must be minimization with convex functions"

    # modify objective to include penalty term
    @objective(jump, Min, objective_function(jump) + output_var)

    return input_var, output_var
end

function forward_pass_ICNN!(jump, input)
    @assert length(input) == length(jump[:z][0, :]) "Incorrect input length."
    [fix(jump[:z][0, neuron], input[neuron]) for neuron in eachindex(input)]
    
    try
        optimize!(jump)
        (last_layer, _) = maximum(keys(jump[:z].data))
        result = value.(jump[:z][last_layer, 1])
        return result
    catch e
        println("Error with forward pass: $e")
        @warn "Input or output outside of bounds or incorrectly constructed model."
        return NaN
    end
end