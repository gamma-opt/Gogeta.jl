"""
Takes a trained EvoTrees model and returns an optimized model with only lazy generated split constraints.

# Parameters
- tree_model: trained `EvoTrees` model to be optimized
- constraint_depth: initial split constraint generation depth from root node
- tree_depth: `EvoTrees` model maximum tree depth - used for finding children to necessary depth

# Output
- opt_model: optimized model where only necessary constraints are generated

"""
function trees_to_relaxed_MIP(tree_model, constraints, tree_depth, objective)
    
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

    if constraints == :createinitial
        for tree in 1:n_trees
            for current_node in 1:(2^(tree_depth - 1))
                if tree_model.trees[tree + 1].split[current_node] == true
                    right_leaves = children(current_node << 1 + 1, leaves[tree])
                    left_leaves = children(current_node << 1, leaves[tree])

                    current_feat, current_splitpoint_index = splits[tree, current_node]

                    @constraint(opt_model, sum(y[tree, leaf] for leaf in right_leaves) <= 1 - x[current_feat, current_splitpoint_index])
                    @constraint(opt_model, sum(y[tree, leaf] for leaf in left_leaves) <= x[current_feat, current_splitpoint_index])
                    
                    initial_constraints += 2
                end
            end
        end
    end
    
    # Objective function (maximize / minimize forest prediction)
    @objective(opt_model, Min, tree_model.trees[1].pred[1] + sum(tree_model.trees[tree + 1].pred[leaves[tree][leaf]] * y[tree, leaf] for tree = 1:n_trees, leaf = 1:n_leaves[tree]))
    if objective == :max
        @objective(opt_model, Max, objective_function(opt_model))
    end

    # Use lazy constraints to generate only needed split constraints
    generated_constraints = 0
    function split_constraint_callback(cb_data)
        
        x_opt = callback_value.(Ref(cb_data), opt_model[:x])
        y_opt = callback_value.(Ref(cb_data), opt_model[:y])

        for tree in 1:n_trees

            current_node = 1 # start investigating from root
        
            while (current_node in leaves[tree]) == false # traverse from root until hitting a leaf
                
                # indices for leaves left/right from current node - indexing based on y vector convention
                right_leaves = children(current_node << 1 + 1, leaves[tree])
                left_leaves = children(current_node << 1, leaves[tree])

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
                        current_node = current_node << 1 # check left child - continue search
                    end
                else # right side chosen...
                    if sum(y_opt[tree, left_leaves]) != 0 # ...but found from left
                        
                        # Add constraint associated with current node (2c constraint)
                        split_cons = @build_constraint(sum(y[tree, left_leaves]) <= x[current_feat, current_splitpoint_index])
                        MOI.submit(opt_model, MOI.LazyConstraint(cb_data), split_cons)
                        generated_constraints += 1
                        return

                    else # ...and found from right
                        current_node = current_node << 1 + 1 # check right child - continue search
                    end
                end

            end
        end
    end

    # Set callback for lazy split constraint generation
    if constraints != :createinitial
        set_attribute(opt_model, MOI.LazyConstraintCallback(), split_constraint_callback)
    end
    optimize!(opt_model)

    println("\nINITIAL CONSTRAINTS: $initial_constraints")
    println("GENERATED CONSTRAINTS: $generated_constraints")

    return get_solution(n_feats, opt_model, n_splits, ordered_splits), objective_value(opt_model), opt_model

end

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
    
    result = Vector{Int64}()
    max = last(leaves)

    function inner(num)
        if num < max
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