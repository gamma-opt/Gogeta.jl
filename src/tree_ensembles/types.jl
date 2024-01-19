using EvoTrees

"""
Universal datatype for storing information about a Tree Ensemble Model.
This is the datatype that is used when creating the integer optimization problem from a tree ensemble.

Different tree models (EvoTrees, XGBoost, RandomForest) require individual conversion functions to this datatype.

# Fields
- `n_trees`: number of trees in the ensemble
- `n_feats`: number of features (input variables) in the model
- `n_leaves`: number of leaves on each tree
- `leaves`: indices of the leaves on each tree
- `splits`: [feature, splitpoint index] pairs accessible by [tree, node]
- `splits_ordered`: splitpoints ordered by split value for each feature
- `n_splits`: number of splitpoints for each feature
- `predictions`: prediction of each node (zero for nodes that are not leaves)
- `split_nodes`: boolean array containing information whether a node is a split node or not

Splitpoints is the set of unique condition values from the ensemble. Each node is associated with a condition value.
"""
struct TEModel
    n_trees::Int64
    n_feats::Int64
    n_leaves::Array{Int64}
    leaves::Array{Array}
    splits::Matrix{Any}
    splits_ordered::Array{Vector}
    n_splits::Array{Int64}
    predictions::Array{Array}
    split_nodes::Array{Array}
    child_leaves::Array{Array}
end

"""
```julia
extract_evotrees_info(evo_model; tree_limit=length(evo_model.trees))
```

Gets the data required for constructing the corresponding MIP from an [EvoTrees](https://github.com/Evovest/EvoTrees.jl) model `evo_model`. 
Returns a custom datatype `TEModel` which contains the necessary information.

# Arguments
- `evo_model`: A trained EvoTrees tree ensemble model.

# Optional arguments
- `tree_limit`: only first *n* trees specified by the argument will be used

"""
function extract_evotrees_info(evo_model; tree_limit=length(evo_model.trees))

    n_trees = tree_limit
    n_feats = length(evo_model.info[:fnames])

    n_leaves = Array{Int64}(undef, n_trees) # number of leaves on each tree
    leaves = Array{Array}(undef, n_trees) # ids of the leaves of each tree

    # Get number of leaves and ids of the leaves on each tree
    for tree in 1:n_trees
        leaves[tree] = findall(node -> evo_model.trees[tree].split[node] == false && (node == 1 || evo_model.trees[tree].split[floor(Int, node / 2)] == true), 1:length(evo_model.trees[tree].split))
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

    return TEModel(n_trees, n_feats, n_leaves, leaves, splits, splits_ordered, n_splits, predictions, split_nodes, Array{Array}(undef, n_trees))

end

"""
Precompute child leaves which are needed for generating the split constraints.
"""
function init_TEModel!(TE::TEModel)

    leaf_dict = Array{Dict}(undef, TE.n_trees)
    [leaf_dict[tree] = Dict([(TE.leaves[tree][leaf], leaf) for leaf in eachindex(TE.leaves[tree])]) for tree in 1:TE.n_trees]

    # pre-compute all children for all active nodes of all trees
    for tree in 1:TE.n_trees
        
        nodes_with_split = findall(split -> split == true, TE.split_nodes[tree])
        TE.child_leaves[tree] = Array{Any}(undef, maximum(TE.leaves[tree]))

        for node in [nodes_with_split; TE.leaves[tree]]
            TE.child_leaves[tree][node] = children(node, leaf_dict[tree], last(TE.leaves[tree]))
        end
    end
end
