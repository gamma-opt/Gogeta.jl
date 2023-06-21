function extract_tree_model_info(tree_model, tree_depth)

    n_trees = length(tree_model.trees) - 1 # number of trees in the model (excluding the bias tree)
    n_feats = length(tree_model.info[:fnames]) # number of features (variables) in the model

    n_leaves = Array{Int64}(undef, n_trees) # array for the number of leaves on each tree
    leaves = Array{Array}(undef, n_trees) # array for the ids of the leaves for each tree

    # Get number of leaves and ids of the leaves on each tree
    for tree in 1:n_trees
        leaves[tree] = findall(x -> x != 0, vec(tree_model.trees[tree + 1].pred))
        n_leaves[tree] = length(leaves[tree])
    end

    splits = Matrix{Any}(undef, n_trees, 2^(tree_depth - 1))
    splits_ordered = Array{Vector}(undef, n_feats)
    n_splits = zeros(Int64, n_feats)
    [splits_ordered[feat] = [] for feat in 1:n_feats]

    for tree in 1:n_trees
        for node in 1:2^(tree_depth - 1)
            if tree_model.trees[tree + 1].split[node] == true
                splits[tree, node] = [tree_model.trees[tree + 1].feat[node], tree_model.trees[tree + 1].cond_float[node]]
                push!(splits_ordered[tree_model.trees[tree + 1].feat[node]], tree_model.trees[tree + 1].cond_float[node]) 
            end
        end
    end
    [unique!(sort!(splits_ordered[feat])) for feat in 1:n_feats]
    [n_splits[feat] = length(splits_ordered[feat]) for feat in 1:n_feats]

    for tree in 1:n_trees
        for node in 1:2^(tree_depth - 1)
            if tree_model.trees[tree + 1].split[node] == true
                
                feature::Int = splits[tree, node][1]
                value = splits[tree, node][2]

                splits[tree, node][2] = searchsortedfirst(splits_ordered[feature], value)

            end
        end
    end

    return n_trees, n_feats, n_leaves, leaves, n_splits, splits, splits_ordered

end

function children(id, leaves)
    
    result::Vector{Int64} = []
    max = last(leaves)

    function inner(num)
        if num <= max
            for leaf_index in eachindex(leaves)
                if num == leaves[leaf_index]
                    push!(result, leaf_index)
                end
            end
            inner(num << 1)
            inner(num << 1 + 1)
        end
    end

    inner(id)

    return result

end

function get_solution(n_feats, model, n_splits, splitpoints)

    smallest_splitpoint = Array{Int64}(undef, n_feats)

    [smallest_splitpoint[feat] = n_splits[feat] + 1 for feat in 1:n_feats]
    for ele in eachindex(model[:x])
        if value(model[:x][ele]) == 1 && ele[2] < smallest_splitpoint[ele[1]]
            smallest_splitpoint[ele[1]] = ele[2]
        end
    end

    solution = Array{Vector}(undef, n_feats)
    for feat in 1:n_feats

        solution[feat] = [-Inf64; Inf64]

        if smallest_splitpoint[feat] <= n_splits[feat]
            solution[feat][2] = splitpoints[feat][smallest_splitpoint[feat]]
        end

        if smallest_splitpoint[feat] > 1
            solution[feat][1] = splitpoints[feat][smallest_splitpoint[feat] - 1]
        end
    end

    return solution
end