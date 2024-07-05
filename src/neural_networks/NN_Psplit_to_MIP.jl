"""
    function NN_formulate_Psplit!(jump_model::JuMP.Model, NN_model::Flux.Chain, P, U_in, L_in; strategy="equalsize", silent=true)

Creates an optimization problem from a `Flux.Chain` model using P-split formulation for the disjunctive constraints.

The parameter P specifies the number of splits

A dummy objective function of 1 is added to the model. The objective is left for the user to define.

# Arguments
- `jump_model`: The constraints and variables will be saved to this optimization model.
- `NN_model`: Neural network model to be formulated.
- `P`: The number of splits
- `init_U`: Upper bounds for the input variables.
- `init_L`: Lower bounds for the input variables.

# Optional arguments
- `strategy`: the way partitioning is done, possible options include: "equalsize", "equalrange", "random". Default is "equalsize".
- `silent`: Controls console ouput.

"""

function NN_formulate_Psplit!(jump_model::JuMP.Model, NN_model::Flux.Chain, P, init_U, init_L; strategy="equalsize", silent=true)

    oldstdout = stdout
    if silent redirect_stdout(devnull) end

    if !(strategy in ["equalrange", "equalsize", "random"])
        throw(ArgumentError("Possible strategy options: \"equalrange\", \"equalsize\", \"random\""))
    end
    if strategy=="equalrange" && P<=2
        throw(ArgumentError("With equalrange strategy number of partitions P>2"))
    end
        
    println("Creating JuMP model...")
    empty!(jump_model)

    K = length(NN_model); # number of layers (input layer not included)
    W = deepcopy([Flux.params(NN_model)[2*k-1] for k in 1:K]);
    b = deepcopy([Flux.params(NN_model)[2*k] for k in 1:K]);


    @assert all([NN_model[i].σ == relu for i in 1:K-1]) "Neural network must use the relu activation function."
    @assert NN_model[K].σ == identity "Neural network must use the identity function for the output layer."
    @assert P > 0 "The number of splits must be more than 0."

    input_length = Int((length(W[1]) / length(b[1])))
    neuron_count = [length(b[k]) for k in eachindex(b)]
    neurons(layer) = layer == 0 ? [i for i in 1:input_length] : [i for i in 1:neuron_count[layer]]

    @variable(jump_model, x[layer = 0:K, neurons(layer)]);
    @variable(jump_model, sigma[layer = 1:K-1, neurons(layer)]);
    @variable(jump_model, z_b[layer = 1:K-1, neurons(layer), p=1:P]);

    @constraint(jump_model, [j = 1:input_length], x[0, j] <= init_U[j])
    @constraint(jump_model, [j = 1:input_length], x[0, j] >= init_L[j])

    bounds_U = Vector{Vector}(undef, K)
    bounds_L = Vector{Vector}(undef, K)

    UB_α = Vector{Vector{Vector}}(undef, K-1)
    LB_α = Vector{Vector{Vector}}(undef, K-1)

    [UB_α[layer] = Vector{Vector}(undef, neuron_count[layer]) for layer in 1:K-1]
    [LB_α[layer] = Vector{Vector}(undef, neuron_count[layer]) for layer in 1:K-1]

    for layer in 1:K 

        println("\nLAYER $layer")

        if layer == 1

            bounds_U[layer] = [sum(max(W[layer][neuron, previous] * init_U[previous], W[layer][neuron, previous] * init_L[previous]) for previous in neurons(layer-1)) + b[layer][neuron] for neuron in neurons(layer)]
            bounds_L[layer] = [sum(min(W[layer][neuron, previous] * init_U[previous], W[layer][neuron, previous] * init_L[previous]) for previous in neurons(layer-1)) + b[layer][neuron] for neuron in neurons(layer)]
        else
            
            bounds_U[layer] = [sum(bounds_U[layer-1][previous]*max(0, W[layer][neuron, previous]) + bounds_L[layer-1][previous]*min(0, W[layer][neuron, previous]) for previous in neurons(layer-1)) + b[layer][neuron] for neuron in neurons(layer)]
            bounds_L[layer] = [sum(bounds_L[layer-1][previous]*max(0, W[layer][neuron, previous]) + bounds_U[layer-1][previous]*min(0, W[layer][neuron, previous]) for previous in neurons(layer-1)) + b[layer][neuron] for neuron in neurons(layer)]
        end

        # output bounds calculated but no more constraints added
        if layer == K
            break
        end

        [UB_α[layer][neuron] = Vector(undef, P) for neuron in 1:neuron_count[layer]]
        [LB_α[layer][neuron] = Vector(undef, P) for neuron in 1:neuron_count[layer]]

        @constraint(jump_model, [neuron in neurons(layer)], x[layer, neuron] <= max(0, bounds_U[layer][neuron]))
        @constraint(jump_model, [neuron in neurons(layer)], x[layer, neuron] >= 0)

        for neuron in neurons(layer)

            split_indices = Psplits(W[layer][neuron, :], P, strategy)
            set_binary(sigma[layer, neuron])
            
            @constraint(jump_model, sum(sum(W[layer][neuron, i]*x[layer-1, i] for i in split_indices[p])-z_b[layer, neuron, p] for p in eachindex(split_indices)) + sigma[layer, neuron]*b[layer][neuron]<=0)
            @constraint(jump_model, sum(z_b[layer, neuron, p] for p in  eachindex(split_indices)) + (1-sigma[layer,neuron])*b[layer][neuron]>=0)
            @constraint(jump_model, x[layer, neuron] == sum(z_b[layer, neuron, p] for p in eachindex(split_indices)) + (1-sigma[layer,neuron])*b[layer][neuron]) 
            
            for p in eachindex(split_indices)
            
                if layer == 1
                    UB_α[layer][neuron][p] = sum(max(W[layer][neuron, previous] * init_U[previous], W[layer][neuron, previous] * init_L[previous]) for previous in split_indices[p])
                    LB_α[layer][neuron][p] = sum(min(W[layer][neuron, previous] * init_U[previous], W[layer][neuron, previous] * init_L[previous]) for previous in split_indices[p])
                else
                    UB_α[layer][neuron][p] = sum(max(W[layer][neuron, previous] * max(0, bounds_U[layer-1][previous]), W[layer][neuron, previous] * max(0, bounds_L[layer-1][previous])) for previous in split_indices[p])
                    LB_α[layer][neuron][p] = sum(min(W[layer][neuron, previous] * max(0, bounds_U[layer-1][previous]), W[layer][neuron, previous] * max(0, bounds_L[layer-1][previous])) for previous in split_indices[p])
                end
                
                @constraint(jump_model, sigma[layer, neuron]*LB_α[layer][neuron][p]<=sum(W[layer][neuron, i]*x[layer-1, i] for i in split_indices[p])-z_b[layer, neuron, p])
                @constraint(jump_model, sum(W[layer][neuron, i]*x[layer-1, i] for i in split_indices[p])-z_b[layer, neuron, p]<=sigma[layer, neuron]*UB_α[layer][neuron][p]) 
                @constraint(jump_model, (1-sigma[layer, neuron])*LB_α[layer][neuron][p]<=z_b[layer, neuron, p])
                @constraint(jump_model, z_b[layer, neuron, p]<=(1-sigma[layer, neuron])*UB_α[layer][neuron][p])

            end
        end
    end

    # output layer
    @constraint(jump_model, [neuron in neurons(K)], x[K, neuron] <= max(0, bounds_U[K][neuron]))
    @constraint(jump_model, [neuron in neurons(K)], x[K, neuron] >= min(0, bounds_L[K][neuron]))
    @constraint(jump_model, [neuron in neurons(K)], x[K, neuron] == b[K][neuron] + sum(W[K][neuron, i] * x[K-1, i] for i in neurons(K-1)))

    #A dummy objective
    @objective(jump_model, Max, 1);

    redirect_stdout(oldstdout)

    return bounds_U, bounds_L
    
end
