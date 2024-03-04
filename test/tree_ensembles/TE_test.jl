using EvoTrees
using JuMP
using GLPK
using JuMP

@info "Loading a trained tree ensemble model."

evo_model = EvoTrees.load("tree_ensembles/paraboloid.bson");

universal_tree_model = extract_evotrees_info(evo_model)

@info "Creating JuMP models and optimizing them."

jump_model_lazy = direct_model(GLPK.Optimizer());
set_silent(jump_model_lazy)

TE_formulate!(jump_model_lazy, universal_tree_model, MIN_SENSE);

function split_constraint_callback(cb_data)

    status = callback_node_status(cb_data, jump_model_lazy)

    # Only run at integer solutions
    if status != MOI.CALLBACK_NODE_STATUS_INTEGER
        return
    end

    tree_callback_algorithm(cb_data, universal_tree_model, jump_model_lazy)
end
set_attribute(jump_model_lazy, MOI.LazyConstraintCallback(), split_constraint_callback) 
optimize!(jump_model_lazy)

jump_model_all = direct_model(GLPK.Optimizer());
set_silent(jump_model_all)

TE_formulate!(jump_model_all, universal_tree_model, MIN_SENSE);
add_split_constraints!(jump_model_all, universal_tree_model)
optimize!(jump_model_all)

@info "Getting solutions."

solution_lazy = get_solution(jump_model_lazy, universal_tree_model)
solution_all = get_solution(jump_model_all, universal_tree_model)

objective_value_lazy = objective_value(jump_model_lazy)
objective_value_all = objective_value(jump_model_all)

# These values have been obtained using exhaustive bruteforce search.
# This is the true global minimum of the tree ensemble.
minimum_value = -0.0008044997f0;
minimum = [ -0.0373748243818212, -0.042113434576107486,  0.009143172249250781];

@info "Comparing obtained optimal solutions to the true minimum of the tree ensemble."

@test solution_lazy ≈ solution_all
@test objective_value_all ≈ objective_value_lazy
@test objective_value_all ≈ minimum_value
@test all([minimum[i] > solution_all[i][1] && minimum[i] < solution_all[i][2] for i in 1:3])
