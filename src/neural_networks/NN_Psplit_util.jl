"""
    Psplits(w::AbstractVector, P::Integer, strategy::String)

Given weight vector `w` depending on partioning strategy splits indexes of elements of `w` to P sets. If some of the sets are empty, they are dropped. 
So, make sure that the number of partitions provided is a meaningful number. 

The partions are formed according to the strategy. Possible strategies include "equalsize", "equalrange", and "random".

Returns P sets of partioned indexes.
"""

function Psplits(w::AbstractVector, P::Integer, strategy::String)

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
      
    end
    if strategy=="equalrange"
        # Define the thresholds v
        v = Vector{Float64}(undef, P + 1)
        v[1] =  minimum(w)
        v[P+1] = maximum(w)
        v[2:P] = collect(range(quantile(w, 0.05), quantile(w, 0.95), P-1))

        # Assign indices to partitions S_n
        for i in 1:P
            parts[i] = []
        end
    
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

        for i in 1:P
            parts[i] = []
        end

        for (index, weight) in enumerate(w)
            i = rand(1:P)
            push!(parts[i], index)
        end

    end
    
    
    parts = [s for s in parts if !all(isempty.(s))]

    return parts
end