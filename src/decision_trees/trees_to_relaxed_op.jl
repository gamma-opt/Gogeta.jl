function trees_to_relaxed_MIP(tree_model, tree_depth; objective, constraints)
    
    creation_time = @elapsed begin
    "Data extraction from tree model"
 
    n_trees, n_feats, n_leaves, leaves, n_splits, splits, ordered_splits = extract_tree_model_info(tree_model, tree_depth)
    
    # set up dictionary for leaves when finding child nodes
    leaf_dict = Array{Any}(undef, n_trees)
    [leaf_dict[tree] = Dict([(leaves[tree][leaf], leaf) for leaf in eachindex(leaves[tree])]) for tree in 1:n_trees]

    # pre-compute all children for all nodes of all trees
    child_leaves = Array{Any}(undef, n_trees)
    for tree in 1:n_trees
        
        child_leaves[tree] = Array{Any}(undef, length(tree_model.trees[tree + 1].split))

        for node in eachindex(child_leaves[tree])
            child_leaves[tree][node] = children(node, leaf_dict[tree], last(leaves[tree]))
        end
    end

    # Set up model
    opt_model = direct_model(Gurobi.Optimizer(ENV))
    set_attribute(opt_model, "OutputFlag", 0)
    set_attribute(opt_model, "Presolve", 0)
    set_attribute(opt_model, "TimeLimit", 100.0)

    # Variable definitions as well as constraints (2g) and (2h)
    @variable(opt_model, x[feat = 1:n_feats, 1:n_splits[feat]], Bin) # indicator variable x_ij for feature i <= j:th split point (2g)
    @variable(opt_model, y[tree = 1:n_trees, 1:n_leaves[tree]] >= 0) # indicator variable y_tl for observation falling on leaf l of tree (2h)

    # Constraints (2f) and (2b) (constraint (2e) concerns only categorical variables)
    @constraint(opt_model, [i = 1:n_feats, j = 1:(n_splits[i]-1)], x[i,j] <= x[i, j+1]) # constraints regarding order of split points (2f)
    @constraint(opt_model, [tree = 1:n_trees], sum(y[tree, leaf] for leaf = 1:n_leaves[tree]) == 1) # observation must fall on exactly one leaf (2b)

    # Constraints (2c) and (2d)
    initial_constraints = 0

    if constraints == "initial"
        for tree in 1:n_trees
            for current_node in findall(s -> s==true, tree_model.trees[tree + 1].split)

                right_leaves = child_leaves[tree][current_node << 1 + 1]
                left_leaves = child_leaves[tree][current_node << 1]

                current_feat, current_splitpoint_index = splits[tree, current_node]

                @constraint(opt_model, sum(y[tree, leaf] for leaf in right_leaves) <= 1 - x[current_feat, current_splitpoint_index])
                @constraint(opt_model, sum(y[tree, leaf] for leaf in left_leaves) <= x[current_feat, current_splitpoint_index])
                
                initial_constraints += 2
            end
        end
    end
    
    # Objective function (maximize / minimize forest prediction)
    @objective(opt_model, Min, tree_model.trees[1].pred[1] + sum(tree_model.trees[tree + 1].pred[leaves[tree][leaf]] * y[tree, leaf] for tree = 1:n_trees, leaf = 1:n_leaves[tree]))
    if objective == "max"
        @objective(opt_model, Max, objective_function(opt_model))
    end
    end
    println("\nTIME SPENT CREATING MODEL: $(round(creation_time, digits=2)) seconds")
    println("\nINITIAL CONSTRAINTS: $initial_constraints")

    # Use lazy constraints to generate only needed split constraints
    generated_constraints = 0
    function split_constraint_callback(cb_data, cb_where::Cint)

        # Only run at integer solutions
        if cb_where != GRB_CB_MIPSOL
            return
        end

        Gurobi.load_callback_variable_primal(cb_data, cb_where)

        for tree in 1:n_trees

            current_node = 1 # start investigating from root
        
            while tree_model.trees[tree + 1].split[current_node] == true # traverse from root until hitting a leaf
                
                right_leaves = child_leaves[tree][current_node << 1 + 1]
                left_leaves = child_leaves[tree][current_node << 1]

                # feature and split point index associated with current node
                current_feat::Int64, current_splitpoint_index::Int64 = round.(splits[tree, current_node])

                if round(callback_value(cb_data, x[current_feat, current_splitpoint_index])) == 1 # node condition true - left side chosen...
                    if sum(round(callback_value(cb_data, y[tree, leaf])) for leaf in right_leaves) > 0 # ...but found from right

                        # Add constraint associated with current node (2d constraint)
                        split_cons = @build_constraint(sum(y[tree, leaf] for leaf in right_leaves) <= 1 - x[current_feat, current_splitpoint_index])
                        MOI.submit(opt_model, MOI.LazyConstraint(cb_data), split_cons)
                        generated_constraints += 1
                        return

                    else # ...and found from left
                        current_node = current_node << 1 # check left child - continue search
                    end
                else # right side chosen...
                    if sum(round(callback_value(cb_data, y[tree, leaf])) for leaf in left_leaves) > 0 #...but found from left
                        
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

    # Set callback for lazy split constraint generation
    if constraints == "generate"
        set_attribute(opt_model, "LazyConstraints", 1)
        set_attribute(opt_model, Gurobi.CallbackFunction(), split_constraint_callback)
    end
    opt_time = @elapsed optimize!(opt_model)

    println("GENERATED CONSTRAINTS: $generated_constraints")

    println("\nTIME SPENT OPTIMIZING: $(round(opt_time, digits=2)) seconds\n")

    if termination_status(opt_model) == MOI.OPTIMAL
        println("SOLVED TO OPTIMALITY: $(objective_value(opt_model))")
        return get_solution(n_feats, opt_model, n_splits, ordered_splits), opt_model
    else
        println("SOLVE FAILED, TIME LIMIT 300s REACHED")
        return nothing, opt_model
    end

end