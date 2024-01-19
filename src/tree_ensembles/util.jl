"""
```julia
function children(id::Int, leaf_dict::Dict, max::Int)
```

Finds the leaf indices of the children leaves of node `id` in a binary tree.

Returns an array of the leaf indices.

# Arguments
- `id`: Index of the node in a binary tree. Indexing starts from one and follows level order.
- `leaf_dict`: A dictionary (map) of the leaf indices accessible by the node indices.
- `max`: Biggest possible node id in the tree. Used to terminate the search.

"""
function children(id::Int, leaf_dict::Dict, max::Int)

    result::Vector{Int} = []

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
```julia
function get_solution(model::JuMP.Model, TE::TEModel)
```

Finds the upper and lower bounds for each input variable given the optimized model.

Returns the bounds for each feature in an array.

# Arguments
- `model`: The optimized JuMP model.
- `TE`: Struct of type `TEModel` containing information about the tree ensemble.

"""
function get_solution(model::JuMP.Model, TE::TEModel)

    smallest_splitpoint = Array{Int64}(undef, TE.n_feats)

    [smallest_splitpoint[feat] = TE.n_splits[feat] + 1 for feat in 1:TE.n_feats]
    for ele in eachindex(model[:x])
        if round(value(model[:x][ele])) == 1 && ele[2] < smallest_splitpoint[ele[1]]
            smallest_splitpoint[ele[1]] = ele[2]
        end
    end

    solution = Array{Vector}(undef, TE.n_feats)
    for feat in 1:TE.n_feats

        solution[feat] = [-Inf64; Inf64]

        if smallest_splitpoint[feat] <= TE.n_splits[feat]
            solution[feat][2] = TE.splits_ordered[feat][smallest_splitpoint[feat]]
        end

        if smallest_splitpoint[feat] > 1
            solution[feat][1] = TE.splits_ordered[feat][smallest_splitpoint[feat] - 1]
        end
    end

    return solution
end