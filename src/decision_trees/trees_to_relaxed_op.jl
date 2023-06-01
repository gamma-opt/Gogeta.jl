"""
Takes a trained EvoTrees model and returns an optimization model with only lazy generated split constraints.

# Parameters
- tree_model: trained `EvoTrees` model to be optimized
- depth: initial split constraint generation depth from root node

# Output
- opt_model: optimized model where only necessary constraints are generated

"""
function trees_to_relaxed_MIP(tree_model, constraint_depth, tree_depth)
    
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

    "Optimization model and constraint generation"

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

    # Use lazy constraints to generate only needed split constraints
    function split_constraint_callback(cb_data)
        
        x_val = callback_value.(Ref(cb_data), x)
        y_val = callback_value.(Ref(cb_data), y)

        for tree in 1:n_trees

            current_node = 1
        
            while current_node in leaves[1] == false
                
                if # left side chosen
                    if sum(findall( leaf -> leaf in children(2*current_node, tree_depth), leaves[tree])) == 0 # not found from left
                        
                        # Add current node constraint
                        split_cons = @build_constraint()
                        MOI.submit(opt_model, MOI.LazyConstraint(cb_data), split_cons)

                    else # found from left
                        current_node *= 2 # check left children constraint
                    end
                else # right side chosen
                    if sum(y_val[tree, children(2*current_node + 1, tree_depth)]) == 0 # not found from right
                        
                        # Add current node constraint
                        split_cons = @build_constraint()
                        MOI.submit(opt_model, MOI.LazyConstraint(cb_data), split_cons)

                    else # found from right
                        current_node = 2*current_node + 1 # check right children constraint
                    end
                end

            end
        end

    end

    MOI.set(opt_model, MOI.LazyConstraintCallback(), split_constraint_callback)

    # Optimize model
    optimize!(opt_model);

    return opt_model

end

function children(id, depth)
    
    result = Vector{Int64}()

    function inner(num)
        if num < 2^depth - 1
            push!(result, num)
            inner(2*num)
            inner(2*num + 1)
        end
    end

    inner(id)

    return result

end