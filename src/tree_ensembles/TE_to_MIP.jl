using JuMP

"""
Solver-independent version of tree ensemble to MIP conversion.
"""
function TE_to_MIP(tree_model, optimizer; create_initial=false, objective=MAX_SENSE, timelimit=100)

    n_trees = tree_model.n_trees
    n_feats = tree_model.n_feats
    n_leaves = tree_model.n_leaves
    leaves = tree_model.leaves
    splits = tree_model.splits
    splits_ordered = tree_model.splits_ordered
    n_splits = tree_model.n_splits
    predictions = tree_model.predictions
    split_nodes = tree_model.split_nodes
    
    # set up dictionary for leaves when finding child nodes
    leaf_dict = Array{Any}(undef, n_trees)
    [leaf_dict[tree] = Dict([(leaves[tree][leaf], leaf) for leaf in eachindex(leaves[tree])]) for tree in 1:n_trees]

    # pre-compute all children for all active nodes of all trees
    child_leaves = Array{Any}(undef, n_trees)
    for tree in 1:n_trees
        
        nodes_with_split = findall(split -> split == true, split_nodes[tree])
        child_leaves[tree] = Array{Any}(undef, maximum(leaves[tree]))

        for node in [nodes_with_split; leaves[tree]]
            child_leaves[tree][node] = children(node, leaf_dict[tree], last(leaves[tree]))
        end
    end

    # Set up model
    opt_model = direct_model(optimizer)
    set_silent(opt_model)
    set_time_limit_sec(opt_model, timelimit)
    solver = solver_name(opt_model)

    # Variable definitions as well as constraints (2g) and (2h)
    @variable(opt_model, x[feat = 1:n_feats, 1:n_splits[feat]], Bin) # indicator variable x_ij for feature i <= j:th split point (2g)
    @variable(opt_model, y[tree = 1:n_trees, 1:n_leaves[tree]] >= 0) # indicator variable y_tl for observation falling on leaf l of tree t (2h)

    # Constraints (2f) and (2b) (constraint (2e) concerns only categorical variables)
    @constraint(opt_model, [i = 1:n_feats, j = 1:(n_splits[i]-1)], x[i,j] <= x[i, j+1]) # constraints regarding order of split points (2f)
    @constraint(opt_model, [tree = 1:n_trees], sum(y[tree, leaf] for leaf = 1:n_leaves[tree]) == 1) # observation must fall on exactly one leaf (2b)

    # Constraints (2c) and (2d)
    initial_constraints = 0

    if create_initial == true
        for tree in 1:n_trees
            for current_node in findall(s -> s==true, split_nodes[tree])

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
    @objective(opt_model, objective, sum(predictions[tree][leaves[tree][leaf]] * y[tree, leaf] for tree = 1:n_trees, leaf = 1:n_leaves[tree]))

    create_initial && println("\nINITIAL CONSTRAINTS: $initial_constraints")

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

    generated_constraints = 0
    function callback_algorithm(cb_data)
        for tree in 1:n_trees

            current_node = 1 # start investigating from root
        
            while split_nodes[tree][current_node] == true # traverse from root until hitting a leaf

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
                        break

                    else # ...and found from left
                        current_node = current_node << 1 # check left child - continue search
                    end
                else # right side chosen...
                    if sum(round(callback_value(cb_data, y[tree, leaf])) for leaf in left_leaves) > 0 #...but found from left
                        
                        # Add constraint associated with current node (2c constraint)
                        split_cons = @build_constraint(sum(y[tree, leaf] for leaf in left_leaves) <= x[current_feat, current_splitpoint_index])
                        MOI.submit(opt_model, MOI.LazyConstraint(cb_data), split_cons)
                        generated_constraints += 1
                        break

                    else # ...and found from right
                        current_node = current_node << 1 + 1 # check right child - continue search
                    end
                end
            end
        end
    end

    # Set callback for lazy split constraint generation
    if create_initial == false
        if solver == "Gurobi"
            set_attribute(opt_model, "LazyConstraints", 1)
            set_attribute(opt_model, Gurobi.CallbackFunction(), split_constraint_callback_gurobi)
        elseif solver == "GLPK"
            set_attribute(opt_model, MOI.LazyConstraintCallback(), split_constraint_callback)
        else
            println("SOLVER NOT SUPPORTED FOR LAZY CONSTRAINTS.")
            return
        end
    end

    opt_time = @elapsed optimize!(opt_model)

    !create_initial && println("\nGENERATED CONSTRAINTS: $generated_constraints")

    println("\nTIME SPENT OPTIMIZING: $(round(opt_time, digits=2)) seconds\n")

    constraints = create_initial == true ? initial_constraints : generated_constraints

    # Print solution
    if termination_status(opt_model) == MOI.OPTIMAL
        println("SOLVED TO OPTIMALITY: $(objective_value(opt_model))")
        return get_solution(n_feats, opt_model, n_splits, splits_ordered), objective_value(opt_model), opt_model, constraints, opt_time
    elseif termination_status(opt_model) == MOI.TIME_LIMIT
        println("SOLVE FAILED, TIME LIMIT REACHED")
    else
        println("SOLVE FAILED")
    end
    return nothing, opt_model, constraints, opt_time
end