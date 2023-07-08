"""
Gets the data required for constructing the corresponding MIP from an [EvoTrees](https://github.com/Evovest/EvoTrees.jl) model `evo_model`. Returns a custom datatype `TEModel` which contains the necessary information.
"""
function extract_evotrees_info(evo_model; tree_limit=length(evo_model.trees))

    n_trees = tree_limit
    n_feats = length(evo_model.info[:fnames])

    n_leaves = Array{Int64}(undef, n_trees) # number of leaves on each tree
    leaves = Array{Array}(undef, n_trees) # ids of the leaves of each tree

    # Get number of leaves and ids of the leaves on each tree
    for tree in 1:n_trees
        leaves[tree] = findall(pred -> pred != 0, vec(evo_model.trees[tree].pred))
        n_leaves[tree] = length(leaves[tree])
    end

    splits = Matrix{Any}(undef, n_trees, length(evo_model.trees[2].split)) # storing the feature number and splitpoint index for each split node
    splits_ordered = Array{Vector}(undef, n_feats) # splitpoints for each feature

    n_splits = zeros(Int64, n_feats)
    [splits_ordered[feat] = [] for feat in 1:n_feats]

    for tree in 1:n_trees
        for node in eachindex(evo_model.trees[tree].split)
            if evo_model.trees[tree].split[node] == true
                splits[tree, node] = [evo_model.trees[tree].feat[node], evo_model.trees[tree].cond_float[node]] # save feature and split value
                push!(splits_ordered[evo_model.trees[tree].feat[node]], evo_model.trees[tree].cond_float[node]) # push split value to splits_ordered
            end
        end
    end
    [unique!(sort!(splits_ordered[feat])) for feat in 1:n_feats] # sort splits_ordered and remove copies
    [n_splits[feat] = length(splits_ordered[feat]) for feat in 1:n_feats] # store number of split points

    for tree in 1:n_trees
        for node in eachindex(evo_model.trees[tree].split)
            if evo_model.trees[tree].split[node] == true
                
                feature::Int = splits[tree, node][1]
                value = splits[tree, node][2]

                splits[tree, node][2] = searchsortedfirst(splits_ordered[feature], value)

            end
        end
    end
    predictions = Array{Array}(undef, n_trees)
    [predictions[tree] = evo_model.trees[tree].pred for tree in 1:n_trees]

    split_nodes = Array{Array}(undef, n_trees)
    [split_nodes[tree] = evo_model.trees[tree].split for tree in 1:n_trees]

    return TEModel(n_trees, n_feats, n_leaves, leaves, splits, splits_ordered, n_splits, predictions, split_nodes)

end

"""
Finds the children leaves of node `id` in a binary tree.
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
Gets human-readable array `solution` where upper and lower bounds for each feature are given.
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