function trees_to_relaxed_MIP(tree_model, constraints, tree_depth, objective)
    
    "Data extraction from tree model"
 
    n_trees, n_feats, n_leaves, leaves, n_splits, splits, ordered_splits = extract_tree_model_info(tree_model, tree_depth)
    
    opt_model = direct_model(Gurobi.Optimizer(ENV))
    set_attribute(opt_model, "OutputFlag", 0)
    set_attribute(opt_model, "Presolve", 0)
    set_attribute(opt_model, "LazyConstraints", 1)

    # Variable definitions as well as constraints (2g) and (2h)
    @variable(opt_model, x[feat = 1:n_feats, 1:n_splits[feat]], Bin) # indicator variable x_ij for feature i <= j:th split point (2g)
    @variable(opt_model, y[tree = 1:n_trees, 1:n_leaves[tree]] >= 0) # indicator variable y_tl for observation falling on leaf l of tree (2h)

    # Constraints (2f) and (2b) (constraint (2e) concerns only categorical variables)
    @constraint(opt_model, [i = 1:n_feats, j = 1:(n_splits[i]-1)], x[i,j] <= x[i, j+1]) # constraints regarding order of split points (2f)
    @constraint(opt_model, [tree = 1:n_trees], sum(y[tree, leaf] for leaf = 1:n_leaves[tree]) == 1) # observation must fall on exactly one leaf (2b)

    # Constraints (2c) and (2d)
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
    function split_constraint_callback(cb_data, cb_where::Cint)

        if cb_where == GRB_CB_MIPSOL

            Gurobi.load_callback_variable_primal(cb_data, cb_where)
            x_opt = callback_value.(cb_data, x)
            y_opt = callback_value.(cb_data, y)
    
            for tree in 1:n_trees
    
                current_node = 1 # start investigating from root
            
                while (current_node in leaves[tree]) == false # traverse from root until hitting a leaf
                    
                    # indices for leaves left/right from current node - indexing based on y vector convention
                    right_leaves = children(current_node << 1 + 1, leaves[tree])
                    left_leaves = children(current_node << 1, leaves[tree])
    
                    # feature and split point index associated with current node
                    current_feat, current_splitpoint_index = splits[tree, current_node]
    
                    if round(x_opt[current_feat, current_splitpoint_index]) == 1 # node condition true - left side chosen...
                        if sum(round(y_opt[tree, leaf]) for leaf in right_leaves) > 0 # ...but found from right
    
                            # Add constraint associated with current node (2d constraint)
                            split_cons = @build_constraint(sum(y[tree, leaf] for leaf in right_leaves) <= 1 - x[current_feat, current_splitpoint_index])
                            MOI.submit(opt_model, MOI.LazyConstraint(cb_data), split_cons)
                            generated_constraints += 1
                            return
    
                        else # ...and found from left
                            current_node = current_node << 1 # check left child - continue search
                        end
                    else # right side chosen...
                        if sum(round(y_opt[tree, leaf]) for leaf in left_leaves) > 0 # ...but found from left
                            
                            # Add constraint associated with current node (2c constraint)
                            split_cons = @build_constraint(sum(y[tree, leaf] for leaf in left_leaves) <= x[current_feat, current_splitpoint_index])
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
    end

    # Set callback for lazy split constraint generation
    if constraints != :createinitial
        set_attribute(opt_model, Gurobi.CallbackFunction(), split_constraint_callback)
    end
    optimize!(opt_model)

    println("\nINITIAL CONSTRAINTS: $initial_constraints")
    println("GENERATED CONSTRAINTS: $generated_constraints")
    println("OPTIMAL OBJECTIVE: $(objective_value(opt_model))")

    return get_solution(n_feats, opt_model, n_splits, ordered_splits), objective_value(opt_model), opt_model

end