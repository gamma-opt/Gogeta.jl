"""
Takes a trained EvoTrees model and returns an optimized model with only lazy generated split constraints.

# Parameters
- tree_model: trained `EvoTrees` model to be optimized
- constraint_depth: initial split constraint generation depth from root node
- tree_depth: `EvoTrees` model maximum tree depth - used for finding children to necessary depth

# Output
- opt_model: optimized model where only necessary constraints are generated

"""
function trees_to_relaxed_MIP(tree_model, create_initial_constraints, tree_depth)
    
    "Data extraction from tree model"
 
    n_trees, n_feats, n_leaves, leaves, n_splits, splits, ordered_splits = extract_tree_model_info(tree_model, tree_depth)
    
    opt_model = Model(Gurobi.Optimizer)

    # Variable definitions as well as constraints (2g) and (2h)
    @variable(opt_model, x[feat = 1:n_feats, 1:n_splits[feat]], Bin) # indicator variable x_ij for feature i <= j:th split point (2g)
    @variable(opt_model, y[tree = 1:n_trees, 1:n_leaves[tree]] >= 0) # indicator variable y_tl for observation falling on leaf l of tree (2h)

    # Constraints (2f) and (2b) (constraint (2e) concerns only categorical variables)
    @constraint(opt_model, [i = 1:n_feats, j = 1:(n_splits[i]-1)], x[i,j] <= x[i, j+1]) # constraints regarding order of split points (2f)
    @constraint(opt_model, [tree = 1:n_trees], sum(y[tree, leaf] for leaf = 1:n_leaves[tree]) == 1) # observation must fall on exactly one leaf (2b)

    # Constraints (2c) and (2d) - generate only if create_initial_constraints == true
    initial_constraints = 0

    if create_initial_constraints
        for tree in 1:n_trees
            for current_node in findall(split->split==true, tree_model.trees[tree + 1].split)
                right_leaves = children(2*current_node + 1, leaves[tree])
                left_leaves = children(2*current_node, leaves[tree])

                current_feat, current_splitpoint_index = splits[tree, current_node]

                @constraint(opt_model, sum(y[tree, right_leaves]) <= 1 - x[current_feat, current_splitpoint_index])
                @constraint(opt_model, sum(y[tree, left_leaves]) <= x[current_feat, current_splitpoint_index])
                
                initial_constraints += 2
            end
        end
    end
    
    # Objective function (maximize / minimize forest prediction)
    @objective(opt_model, Min, tree_model.trees[1].pred[1] + sum(tree_model.trees[tree + 1].pred[leaves[tree][leaf]] * y[tree, leaf] for tree = 1:n_trees, leaf = 1:n_leaves[tree]))

    # Use lazy constraints to generate only needed split constraints
    generated_constraints = 0
    function split_constraint_callback(cb_data)
        
        x_opt = callback_value.(Ref(cb_data), opt_model[:x])
        y_opt = callback_value.(Ref(cb_data), opt_model[:y])

        for tree in 1:n_trees

            current_node = 1 # start investigating from root
        
            while (current_node in leaves[tree]) == false # traverse from root until hitting a leaf
                
                # indices for leaves left/right from current node - indexing based on y vector convention
                right_leaves = children(2*current_node + 1, leaves[tree])
                left_leaves = children(2*current_node, leaves[tree])

                # feature and split point index associated with current node
                current_feat, current_splitpoint_index = splits[tree, current_node]

                if x_opt[current_feat, current_splitpoint_index] == 1 # node condition true - left side chosen...
                    if sum(y_opt[tree, right_leaves]) != 0 # ...but found from right

                        # Add constraint associated with current node (2d constraint)
                        split_cons = @build_constraint(sum(y[tree, right_leaves]) <= 1 - x[current_feat, current_splitpoint_index])
                        MOI.submit(opt_model, MOI.LazyConstraint(cb_data), split_cons)
                        generated_constraints += 1
                        return

                    else # ...and found from left
                        current_node = 2*current_node # check left child - continue search
                    end
                else # right side chosen...
                    if sum(y_opt[tree, left_leaves]) != 0 # ...but found from left
                        
                        # Add constraint associated with current node (2c constraint)
                        split_cons = @build_constraint(sum(y[tree, left_leaves]) <= x[current_feat, current_splitpoint_index])
                        MOI.submit(opt_model, MOI.LazyConstraint(cb_data), split_cons)
                        generated_constraints += 1
                        return

                    else # ...and found from right
                        current_node = 2*current_node + 1 # check right child - continue search
                    end
                end

            end
        end
    end

    # Set callback for lazy split constraint generation
    if create_initial_constraints == false
        MOI.set(opt_model, MOI.LazyConstraintCallback(), split_constraint_callback)
    end
    optimize!(opt_model)

    println("\nINITIAL CONSTRAINTS: $initial_constraints")
    println("GENERATED CONSTRAINTS: $generated_constraints")

    return get_solution(n_feats, opt_model, n_splits, ordered_splits), objective_value(opt_model)

end

function extract_tree_model_info(tree_model, tree_depth)

    n_trees = length(tree_model.trees) - 1 # number of trees in the model
    n_feats = length(tree_model.info[:fnames]) # number of features (variables) in the model

    n_leaves = Array{Int64}(undef, n_trees) # array for the number of leaves on each tree
    leaves = Array{Array}(undef, n_trees) # array for the ids of the leaves for each tree

    # Get number of leaves and ids of the leaves on each tree
    for tree in 1:n_trees
        leaves[tree] = findall(x -> x == false, vec(tree_model.trees[tree + 1].split))
        n_leaves[tree] = length(leaves[tree])
    end

    n_splits = zeros(Int64, n_feats) # number of splits for each variable
    splits = Matrix{Any}(undef, n_trees, 2^(tree_depth - 1)) # array of (feature, splitpoint number) indexed by [tree, node]
    ordered_splits = Array{Any}(undef, n_feats)

    # Get number of splits and unique split points for each feature (variable)
    for feat in 1:n_feats
        
        split_tree = Vector{Int64}() # array for trees the split happens in
        split_id = Vector{Int64}() # array for nodes the split happens in
        split_value = Vector{Float64}() # array for split values

        for tree in 1:n_trees

            split_ids = findall(id -> id == feat, tree_model.trees[tree + 1].feat) # nodes with split on feat
            n_splits[feat] += length(split_ids)
            
            # Add node data to splitspoints
            append!(split_id, split_ids)
            append!(split_value, tree_model.trees[tree + 1].cond_float[split_ids])
            append!(split_tree, tree .* ones(length(split_ids)))

        end

        ordered_splits[feat] = hcat(split_tree, split_id, split_value) # save split point data in a matrix
        ordered_splits[feat] = ordered_splits[feat][sortperm(ordered_splits[feat][:, 3]), :] # sort the matrix columns based on the 3rd column (splits values)

        row_num = 0
        for point in eachrow(ordered_splits[feat])
            row_num += 1
            splits[round.(Int, point[1]), round.(Int, point[2])] = feat, row_num
        end

    end

    return n_trees, n_feats, n_leaves, leaves, n_splits, splits, ordered_splits

end

function children(id, leaves)
    
    result = Vector{Int64}()
    max = last(leaves)

    function inner(num)
        if num < max
            for leaf_index in eachindex(leaves)
                if num == leaves[leaf_index]
                    push!(result, leaf_index)
                end
            end
            inner(2*num)
            inner(2*num + 1)
        end
    end

    inner(id)

    return result

end

function get_solution(n_feats, model, n_splits, splitpoints)

    smallest_splitpoint = Array{Int64}(undef, n_feats)
    solution = Array{Float64}(undef, n_feats)

    [smallest_splitpoint[feat] = n_splits[feat] + 1 for feat in 1:n_feats]
    for ele in eachindex(model[:x])
        if value(model[:x][ele]) == 1 && ele[2] < smallest_splitpoint[ele[1]]
            smallest_splitpoint[ele[1]] = ele[2]
        end
    end

    solution = Array{Float32}(undef, n_feats)
    for feat in 1:n_feats
        if smallest_splitpoint[feat] <= n_splits[feat]
            solution[feat] = splitpoints[feat][smallest_splitpoint[feat], 3]
        else
            solution[feat] = Inf32
        end
    end

    return solution
end