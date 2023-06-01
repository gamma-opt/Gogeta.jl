"""
Takes a trained EvoTrees model and returns an optimization model with only lazy generated split constraints.

# Parameters
- tree_model: trained `EvoTrees` model to be optimized
- depth: initial split constraint generation depth from root node

# Output
- opt_model: optimization model with necessary constraints generated

"""
function trees_to_relaxed_MIP(tree_model, depth)
    
    "Data extraction from tree model"

    n_trees = length(tree_model.trees) - 1 # number of trees in the model
    n_feats = length(evo_model.info[:fnames]) # number of features (variables) in the model

    n_leaves = Array{Int64}(undef, n_trees) # array for the number of leaves on each tree
    leaves = Array{Array}(undef, n_trees) # array for the ids of the leaves for each tree

    # Get number of leaves and ids of the leaves on each tree
    for tree in 1:n_trees
        leaves[tree] = findall(x -> x!=0, vec(evo_model.trees[tree + 1].pred))
        n_leaves[tree] = length(leaves[tree])
    end

    n_splits = Array{Int64}(undef, n_feats) # number of splits for each variable
    splitpoints = Array{Any}(undef, n_feats) # ordered list of matrices of unique split points for each feature [tree, node_id, split_value]

    # Get number of splits and unique split points for each feature (variable)
    for feat in 1:n_feats
        
        split_tree = Vector{Int64}() # array for trees the split happens in
        split_id = Vector{Int64}() # array for nodes the split happens in
        split_value = Vector{Float64}() # array for split values

        for tree in 1:n_trees

            split_ids = findall(id -> id == feat, tree_model.trees[tree + 1].feat) # nodes with split on feat
            
            # Add node data to splitspoints
            append!(split_id, split_ids)
            append!(split_value, evo_model.trees[tree + 1].cond_float[split_ids])
            append!(split_tree, tree .* ones(length(split_ids)))

        end

        splitpoints[feat] = [split_tree'; split_id'; split_value'] # save split point data in a matrix
        splitpoints[feat] = splitpoints[feat][:,sortperm(splitpoints[feat][3,:])] # sort the matix columns based on the 3rd column (splits values)
        n_splits[feat] = size(splitpoints[feat], 2) # store the number of splits in the whole forest associated with feat 

    end

    "Optimization model and constraint creation"

    opt_model = Model(Gurobi.Optimizer)

    # Variable definitions as well as constraints (2g) and (2h)
    @variable(opt_model, x[feat = 1:n_feats, 1:n_splits[feat]], Bin) # indicator variable x_ij for feature i <= j:th split point (2g)
    @variable(opt_model, y[tree = 1:n_trees, 1:n_leaves[tree]] >= 0) # indicator variable y_tl for observation falling on leaf l of tree (2h)

    # Constraints (2f) and (2b) (constaint (2e) concerns only categorical variables)
    @constraint(opt_model, [i = 1:n_feats, j = 1:n_splits[i]-1], x[i,j] <= x[i, j+1]) # constraints regarding order of split points (2f)
    @constraint(opt_model, [t = 1:n_trees], sum(y[t,l] for l = 1:n_leaves[t]) == 1) # observation must fall on exactly one leaf (2b)

    # Constraints (2c) and (2d)

    # Objective function (maximize / minimize forest prediction)
    @objective(opt_model, Max, sum(1/n_trees * evo_model.trees[tree + 1].pred[leaves[tree][leaf]] * y[tree, leaf] for tree = 1:n_trees, leaf = 1:n_leaves[tree]))

    # Optimize model
    @time optimize!(opt_model)

    # Print optimal solution
    for feat = 1:n_feats 
        x_opt = Array{Float64}(undef,  n_splits[feat])
        [x_opt[j] = value.(opt_model[:x])[feat, j] for j = 1:n_splits[feat]]
        first_index = findfirst(x -> x==1, x_opt)
        print("x_$feat <= $(splitpoints[feat][3, first_index]) \n")
    end

    return opt_model

end