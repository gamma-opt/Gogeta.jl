"""
Universal datatype for storing information about a Tree Ensemble Model.

# Fields
* `n_trees` - number of trees in the ensemble
* `n_feats` - number of features (variables) in the model
* `n_leaves` - number of leaves on each tree
* `leaves` - ids of the leaves on each tree
* `splits` - [feature, splitpoint index] pairs accessible by [tree, node]
* `splits_ordered` - splitpoints ordered by split value for each feature
* `n_splits` - number of splitpoints for each feature
* `predictions` - prediction of each node (zero for nodes that are not leaves)
* `split_nodes` - boolean array containing information whether a node is a split node or not
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
end