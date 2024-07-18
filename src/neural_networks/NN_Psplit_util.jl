"""
Psplits(w::AbstractVector, P::Integer, strategy::String)

Given weight vector `w` depending on partioning strategy splits indexes of elements of `w` to P sets. If some of the sets are empty, they are dropped. 
So, make sure that the number of partitions provided is a meaningful number. 

The partions are formed according to the strategy. Possible strategies include "equalsize", "equalrange", "snake" and "random".

Returns P sets of partioned indexes.
"""
function Psplits(w::AbstractVector, P::Integer, strategy::String; silent=true)

    parts = Vector{Vector{Int64}}(undef, P)

    if strategy=="equalsize"

        X = sortperm(w)
        l = length(X)
        n = l % P

        s2 = (l÷P)

        if n > 0
            s1 = (l÷P)+1
            for k in 1:n
                parts[k] = X[1+(k-1)*s1:(k-1)*s1+s1]
            end
        else
            s1 = 0
        end
        
        for k in n+1:P
            parts[k] = X[1+n*s1+(k-1-n)*s2:n*s1+(k-1-n)*s2+s2]
        end
    else

        for i in 1:P
            parts[i] = []
        end

        if strategy=="equalrange"
            # Define the thresholds v
            v = Vector{Float64}(undef, P + 1)
            v[1] =  minimum(w)
            v[P+1] = maximum(w)
            v[2:P] = collect(range(quantile(w, 0.05), quantile(w, 0.95), P-1))

            for (index, weight) in enumerate(w)
                for i in 1:P
                    if weight >= v[i] && weight <= v[i+1]
                        push!(parts[i], index)
                        break
                    end
                end
            end
        end

        if strategy=="random"

            i = rand(1:P, length(w))
            for (index, weight) in enumerate(w)
                push!(parts[i[index]], index)
            end

        end

        if strategy=="snake"

            X = sortperm(w)
            for (i, j) in enumerate(X)
                if (div(i-1, P) % 2 == 1)
                    push!(parts[P - (i-1) % P], j)
                else
                    push!(parts[(i-1) % P + 1], j)
                end
            end
        end
    end
    if silent==false
        
        init_len = length(vcat(parts))
        parts = [s for s in parts if !all(isempty.(s))]
        new_len = length(vcat(parts))
        if init_len>new_len
            println("Warning: Empty partitions were dropped. Try decreasing number of partitions or changing strategy.")
        end

    else
        parts = [s for s in parts if !all(isempty.(s))]
    end

    return parts
end

"""
function calculate_bounds_α(model::JuMP.Model, layer, neuron, W, split_indices)

Function that finds tighier bounds for partioned neuronss

"""


function calculate_bounds_α(model::JuMP.Model, layer, neuron, W, split_indices)

    @objective(model, Max, sum(W[layer][neuron, i] * model[:x][layer-1, i] for i in split_indices))
    optimize!(model)
    
    upper_bound = if termination_status(model) == OPTIMAL
        objective_value(model)
    else
        @warn "Layer $layer, neuron $neuron could not be solved to optimality."
        objective_bound(model)
    end

    set_objective_sense(model, MIN_SENSE)
    optimize!(model)
 
    lower_bound = if termination_status(model) == OPTIMAL
        objective_value(model)
    else
        @warn "Layer $layer, neuron $neuron could not be solved to optimality."
        objective_bound(model)
    end


    return upper_bound, lower_bound
end