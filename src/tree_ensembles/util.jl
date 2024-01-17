"""
```julia
children(id, leaf_dict, max)
```

Finds the leaf indices of the children leaves of node `id` in a binary tree.

Returns an array of the leaf indices.

# Arguments
- `id`: Index of the node in a binary tree. Indexing starts from one and follows level order.
- `leaf_dict`: A dictionary (map) of the leaf indices accessible by the node indices.
- `max`: Biggest possible node id in the tree. Used to terminate the search.

"""
function children(id, leaf_dict, max)

    result::Vector{Int64} = []

    function inner(num)
        if num <= max
            leaf_index = get(leaf_dict, num, 0)
            if leaf_index != 0
                push!(result, leaf_index)
            end
            inner(num << 1)
            inner(num << 1 + 1)
        end
    end

    inner(id)

    return result

end

"""
Creates human-readable array `solution` where upper and lower bounds for each input variable are given.
"""

"""
```julia
get_solution(n_feats, model, n_splits, splits_ordered)
```

Finds the upper and lower bounds for each input variable are given for the optimal solution.

Return the bounds in an array for each feature.

# Arguments
- `n_feats`: Number of features (input variables) in the model.
- `model`: The optimized JuMP model.
- `n_splits`: Number of splits for each variable.
- `splits_ordered`: Ordered list of the splitpoints for each variable.

"""
function get_solution(n_feats, model, n_splits, splits_ordered)

    smallest_splitpoint = Array{Int64}(undef, n_feats)

    [smallest_splitpoint[feat] = n_splits[feat] + 1 for feat in 1:n_feats]
    for ele in eachindex(model[:x])
        if round(value(model[:x][ele])) == 1 && ele[2] < smallest_splitpoint[ele[1]]
            smallest_splitpoint[ele[1]] = ele[2]
        end
    end

    solution = Array{Vector}(undef, n_feats)
    for feat in 1:n_feats

        solution[feat] = [-Inf64; Inf64]

        if smallest_splitpoint[feat] <= n_splits[feat]
            solution[feat][2] = splits_ordered[feat][smallest_splitpoint[feat]]
        end

        if smallest_splitpoint[feat] > 1
            solution[feat][1] = splits_ordered[feat][smallest_splitpoint[feat] - 1]
        end
    end

    return solution
end