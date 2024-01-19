using JuMP

"""
Solver-independent version of tree ensemble to MIP conversion.
"""
function TE_to_MIP(TE::TEModel, optimizer, objective=MAX_SENSE)
    
    # Precalculate child leaves of every node (needed for creating the split constraints)
    init_TEModel!(TE)

    # Set up model
    opt_model = direct_model(optimizer)

    # Variable definitions as well as constraints (2g) and (2h)
    @variable(opt_model, x[feat = 1:TE.n_feats, 1:TE.n_splits[feat]], Bin) # indicator variable x_ij for feature i <= j:th split point (2g)
    @variable(opt_model, y[tree = 1:TE.n_trees, 1:TE.n_leaves[tree]] >= 0) # indicator variable y_tl for observation falling on leaf l of tree t (2h)

    # Constraints (2f) and (2b) (constraint (2e) concerns only categorical variables)
    @constraint(opt_model, [i = 1:TE.n_feats, j = 1:(TE.n_splits[i]-1)], x[i,j] <= x[i, j+1]) # constraints regarding order of split points (2f)
    @constraint(opt_model, [tree = 1:TE.n_trees], sum(y[tree, leaf] for leaf = 1:TE.n_leaves[tree]) == 1) # observation must fall on exactly one leaf (2b)
    
    # Objective function (maximize / minimize forest prediction)
    @objective(opt_model, objective, sum(TE.predictions[tree][TE.leaves[tree][leaf]] * y[tree, leaf] for tree = 1:TE.n_trees, leaf = 1:TE.n_leaves[tree]))

    return opt_model
end

"""
Create all possible split constraints and optimize the model.
"""
function optimize_with_initial_constraints!(opt_model::JuMP.Model, TE::TEModel)

    # Constraints (2c) and (2d) (split constraints)
    for tree in 1:TE.n_trees
        for current_node in findall(s -> s==true, TE.split_nodes[tree])

            right_leaves = TE.child_leaves[tree][current_node << 1 + 1]
            left_leaves = TE.child_leaves[tree][current_node << 1]

            current_feat, current_splitpoint_index = TE.splits[tree, current_node]

            @constraint(opt_model, sum(opt_model[:y][tree, leaf] for leaf in right_leaves) <= 1 - opt_model[:x][current_feat, current_splitpoint_index])
            @constraint(opt_model, sum(opt_model[:y][tree, leaf] for leaf in left_leaves) <= opt_model[:x][current_feat, current_splitpoint_index])
        end
    end

    optimize!(opt_model)
end

"""
Optimize the model using lazy constraint callback for creating the split constraints.
"""
function optimize_with_lazy_constraints!(opt_model::JuMP.Model, TE::TEModel)

    # Use lazy constraints to generate only needed split constraints
    # two versions of the callback since Gurobi.jl doesn't fully support the JuMP API

    function split_constraint_callback(cb_data)

        status = callback_node_status(cb_data, opt_model)

        # Only run at integer solutions
        if status != MOI.CALLBACK_NODE_STATUS_INTEGER
            return
        end

        callback_algorithm(cb_data)
    end

    function split_constraint_callback_gurobi(cb_data, cb_where::Cint)

        # Only run at integer solutions
        if cb_where != GRB_CB_MIPSOL
            return
        end

        Gurobi.load_callback_variable_primal(cb_data, cb_where)
        callback_algorithm(cb_data)

    end

    function callback_algorithm(cb_data)
        for tree in 1:TE.n_trees
    
            current_node = 1 # start investigating from root
        
            while TE.split_nodes[tree][current_node] == true # traverse from root until hitting a leaf
    
                right_leaves = TE.child_leaves[tree][current_node << 1 + 1]
                left_leaves = TE.child_leaves[tree][current_node << 1]
    
                # feature and split point index associated with current node
                current_feat::Int64, current_splitpoint_index::Int64 = round.(TE.splits[tree, current_node])
    
                if round(callback_value(cb_data, opt_model[:x][current_feat, current_splitpoint_index])) == 1 # node condition true - left side chosen...
                    if sum(round(callback_value(cb_data, opt_model[:y][tree, leaf])) for leaf in right_leaves) > 0 # ...but found from right
    
                        # Add constraint associated with current node (2d constraint)
                        split_cons = @build_constraint(sum(opt_model[:y][tree, leaf] for leaf in right_leaves) <= 1 - opt_model[:x][current_feat, current_splitpoint_index])
                        MOI.submit(opt_model, MOI.LazyConstraint(cb_data), split_cons)
                        break
    
                    else # ...and found from left
                        current_node = current_node << 1 # check left child - continue search
                    end
                else # right side chosen...
                    if sum(round(callback_value(cb_data, opt_model[:y][tree, leaf])) for leaf in left_leaves) > 0 #...but found from left
                        
                        # Add constraint associated with current node (2c constraint)
                        split_cons = @build_constraint(sum(opt_model[:y][tree, leaf] for leaf in left_leaves) <= opt_model[:x][current_feat, current_splitpoint_index])
                        MOI.submit(opt_model, MOI.LazyConstraint(cb_data), split_cons)
                        break
    
                    else # ...and found from right
                        current_node = current_node << 1 + 1 # check right child - continue search
                    end
                end
            end
        end
    end

    # Set callback for lazy split constraint generation
    if solver_name(opt_model) == "Gurobi"
        set_attribute(opt_model, "LazyConstraints", 1)
        set_attribute(opt_model, Gurobi.CallbackFunction(), split_constraint_callback_gurobi)
    elseif solver_name(opt_model) == "GLPK"
        set_attribute(opt_model, MOI.LazyConstraintCallback(), split_constraint_callback)
    else
        println("SOLVER NOT SUPPORTED FOR LAZY CONSTRAINTS.")
        return
    end

    optimize!(opt_model)

end