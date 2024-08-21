"""
    function NN_formulate_Psplit!(jump_model::JuMP.Model, NN_model::Flux.Chain, P, U_in, L_in; silent=true)

Creates an optimization problem from a `Flux.Chain` model using P-split formulation for the disjunctive constraints.

The parameter P specifies the number of splits

A dummy objective function of 1 is added to the model. The objective is left for the user to define.

# Arguments
- `jump_model`: The constraints and variables will be saved to this optimization model.
- `NN_model`: Neural network model to be formulated.
- `P`: The number of splits
- `U_in`: Upper bounds for the input variables.
- `L_in`: Lower bounds for the input variables.

# Optional arguments
- `strategy`: Controls the partioning strategy. Available options are `equalsize` (default), `equalrange`, `random`, `snake`
- `bound_tightening`: How the bounds for neurons are produced. Available options are `fast`(default), `standard`, `precomputed`
- `parallel`: Is used in standard bounding, for speeding up the formulation default is `false`
- `U_bounds`, `L_bounds`: Upper and lower bounds used in only for `precomputed` bound-tightening. 
- `silent`: Controls console ouput. Default is true.

"""
function NN_formulate_Psplit!(jump_model::JuMP.Model, NN_model::Flux.Chain, P, U_in, L_in; strategy="equalsize", bound_tightening="fast", parallel=false, U_bounds=nothing, L_bounds=nothing, silent=true)

    oldstdout = stdout
    if silent redirect_stdout(devnull) end

    @assert bound_tightening in ("precomputed", "fast", "standard") "Accepted bound tightening modes are: precomputed, fast, standard, output."

    if bound_tightening == "precomputed" @assert !isnothing(U_bounds) && !isnothing(L_bounds) "Precomputed bounds must be provided." end

    @assert strategy in ["equalrange", "equalsize", "random", "snake"] "Possible strategy options: \"equalrange\", \"equalsize\", \"random\", \"snake\""
    @assert !(strategy == "equalrange" && P <= 2) "With equalrange strategy number of partitions P must be > 2"
        
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
    @assert input_length == length(U_in) == length(L_in) "Initial bounds arrays must be the same length as the input layer"

    @variable(jump_model, x[layer = 0:K, neurons(layer)]);
    @variable(jump_model, Σ[layer = 1:K-1, neurons(layer)]);
    @variable(jump_model, z_b[layer = 1:K-1, neurons(layer), p=1:P]);

    @constraint(jump_model, [j = 1:input_length], x[0, j] <= U_in[j])
    @constraint(jump_model, [j = 1:input_length], x[0, j] >= L_in[j])

    if bound_tightening != "precomputed" 
        U_bounds = Vector{Vector}(undef, K)
        L_bounds = Vector{Vector}(undef, K)
    end

    UB_α = Vector{Vector{Vector}}(undef, K-1)
    LB_α = Vector{Vector{Vector}}(undef, K-1)
    [UB_α[layer] = Vector{Vector}(undef, neuron_count[layer]) for layer in 1:K-1]
    [LB_α[layer] = Vector{Vector}(undef, neuron_count[layer]) for layer in 1:K-1]

    for layer in 1:K 

        println("\nLAYER $layer")

        if bound_tightening != "precomputed"

            if layer  == 1
                U_bounds[layer] = [sum(max(W[layer][neuron, previous] * U_in[previous], W[layer][neuron, previous] * L_in[previous]) for previous in neurons(layer-1)) + b[layer][neuron] for neuron in neurons(layer)]
                L_bounds[layer] = [sum(min(W[layer][neuron, previous] * U_in[previous], W[layer][neuron, previous] * L_in[previous]) for previous in neurons(layer-1)) + b[layer][neuron] for neuron in neurons(layer)]
            else
                U_bounds[layer] = [sum(max(W[layer][neuron, previous] * max(0, U_bounds[layer-1][previous]), W[layer][neuron, previous] * max(0, L_bounds[layer-1][previous])) for previous in neurons(layer-1)) + b[layer][neuron] for neuron in neurons(layer)]
                L_bounds[layer] = [sum(min(W[layer][neuron, previous] * max(0, U_bounds[layer-1][previous]), W[layer][neuron, previous] * max(0, L_bounds[layer-1][previous])) for previous in neurons(layer-1)) + b[layer][neuron] for neuron in neurons(layer)]
            end

            # compute tighter bounds
            if bound_tightening == "standard"
                
                bounds = if parallel == true # multiprocessing enabled
                    pmap(neuron -> calculate_bounds(copy_model(jump_model), layer, neuron, W[layer], b[layer], neurons), neurons(layer))
                else
                    map(neuron -> calculate_bounds(jump_model, layer, neuron, W[layer], b[layer], neurons), neurons(layer))
                end
                println()
                U_bounds[layer] = min.(U_bounds[layer], [bound[1] for bound in bounds])
                L_bounds[layer] = max.(L_bounds[layer], [bound[2] for bound in bounds])

            end
        end
        
        # output bounds calculated but no more constraints added
        if layer == K
            break
        end

        [UB_α[layer][neuron] = Vector(undef, P) for neuron in 1:neuron_count[layer]]
        [LB_α[layer][neuron] = Vector(undef, P) for neuron in 1:neuron_count[layer]]

        @constraint(jump_model, [neuron in neurons(layer)], x[layer, neuron] <= max(0, U_bounds[layer][neuron]))
        @constraint(jump_model, [neuron in neurons(layer)], x[layer, neuron] >= 0)

        for neuron in neurons(layer)

            split_indices = Psplits(W[layer][neuron, :], P, strategy, silent=silent)
            set_binary(Σ[layer, neuron])
            @constraint(jump_model, sum(sum(W[layer][neuron, i]*x[layer-1, i] for i in split_indices[p])-z_b[layer, neuron, p] for p in eachindex(split_indices)) + Σ[layer, neuron]*b[layer][neuron]<=0)
            @constraint(jump_model, sum(z_b[layer, neuron, p] for p in  eachindex(split_indices)) + (1-Σ[layer,neuron])*b[layer][neuron]>=0)
            @constraint(jump_model, x[layer, neuron] == sum(z_b[layer, neuron, p] for p in eachindex(split_indices)) + (1-Σ[layer,neuron])*b[layer][neuron]) 
    
            
            for p in eachindex(split_indices)
            
                if layer == 1
                    UB_α[layer][neuron][p] = sum(max(W[layer][neuron, previous] * U_in[previous], W[layer][neuron, previous] * L_in[previous]) for previous in split_indices[p])
                    LB_α[layer][neuron][p] = sum(min(W[layer][neuron, previous] * U_in[previous], W[layer][neuron, previous] * L_in[previous]) for previous in split_indices[p])
                else
                    UB_α[layer][neuron][p] = sum(max(W[layer][neuron, previous] * max(0, U_bounds[layer-1][previous]), W[layer][neuron, previous] * max(0, L_bounds[layer-1][previous])) for previous in split_indices[p])
                    LB_α[layer][neuron][p] = sum(min(W[layer][neuron, previous] * max(0, U_bounds[layer-1][previous]), W[layer][neuron, previous] * max(0, L_bounds[layer-1][previous])) for previous in split_indices[p])
                end

                if bound_tightening == "standard"
                
                    bounds = calculate_bounds_α(jump_model, layer, neuron, W, split_indices[p])
                    UB_α[layer][neuron][p] = min.(UB_α[layer][neuron][p], bounds[1])
                    LB_α[layer][neuron][p] = max.(LB_α[layer][neuron][p], bounds[2])
    
                end
                
                @constraint(jump_model, Σ[layer, neuron]*LB_α[layer][neuron][p]<=sum(W[layer][neuron, i]*x[layer-1, i] for i in split_indices[p])-z_b[layer, neuron, p])
                @constraint(jump_model, sum(W[layer][neuron, i]*x[layer-1, i] for i in split_indices[p])-z_b[layer, neuron, p]<=Σ[layer, neuron]*UB_α[layer][neuron][p]) 
                @constraint(jump_model, (1-Σ[layer, neuron])*LB_α[layer][neuron][p]<=z_b[layer, neuron, p])
                @constraint(jump_model, z_b[layer, neuron, p]<=(1-Σ[layer, neuron])*UB_α[layer][neuron][p])

            end
        end
    end

    # output layer
    @constraint(jump_model, [neuron in neurons(K)], x[K, neuron] <= max(0, U_bounds[K][neuron]))
    @constraint(jump_model, [neuron in neurons(K)], x[K, neuron] >= min(0, L_bounds[K][neuron]))
    @constraint(jump_model, [neuron in neurons(K)], x[K, neuron] == b[K][neuron] + sum(W[K][neuron, i] * x[K-1, i] for i in neurons(K-1)))

    #A dummy objective
    @objective(jump_model, Max, 1);

    redirect_stdout(oldstdout)

    if bound_tightening!="precomputed"
        return U_bounds, L_bounds
    end
    
end