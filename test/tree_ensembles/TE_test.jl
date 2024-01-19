using EvoTrees
using JuMP
using GLPK

evo_model = EvoTrees.load("tree_ensembles/paraboloid.bson");

universal_tree_model = extract_evotrees_info(evo_model)

jump_model_lazy = TE_to_MIP(universal_tree_model, GLPK.Optimizer(), MIN_SENSE)
optimize_with_lazy_constraints!(jump_model_lazy, universal_tree_model)

jump_model_all = TE_to_MIP(universal_tree_model, GLPK.Optimizer(), MIN_SENSE)
optimize_with_initial_constraints!(jump_model_all, universal_tree_model)

solution_lazy = get_solution(jump_model_lazy, universal_tree_model)
solution_all = get_solution(jump_model_all, universal_tree_model)

objective_value_lazy = objective_value(jump_model_lazy)
objective_value_all = objective_value(jump_model_all)

# These values have been obtained using exhaustive bruteforce search.
# This is the true global minimum of the tree ensemble.
minimum_value = -0.0008044997f0;
minimum = [ -0.0373748243818212, -0.042113434576107486,  0.009143172249250781];

@test solution_lazy ≈ solution_all
@test objective_value_all ≈ objective_value_lazy
@test objective_value_all ≈ minimum_value
@test reduce(&, [minimum[i] > solution_all[i][1] && minimum[i] < solution_all[i][2] for i in 1:3])
