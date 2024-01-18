using EvoTrees
using JuMP
using GLPK

evo_model = EvoTrees.load("tree_ensembles/paraboloid.bson");

universal_tree_model = extract_evotrees_info(evo_model)

solution_lazy, objective_value_lazy, jump_model_lazy = TE_to_MIP(universal_tree_model, GLPK.Optimizer(); objective=MIN_SENSE, create_initial = false)
solution_all, objective_value_all, jump_model_all = TE_to_MIP(universal_tree_model, GLPK.Optimizer(); objective=MIN_SENSE, create_initial = true)

# These values have been obtained using exhaustive bruteforce search.
# This is the true global minimum of the tree ensemble.
minimum_value = -0.0008044997f0;
minimum = [ -0.0373748243818212, -0.042113434576107486,  0.009143172249250781];

@test solution_lazy ≈ solution_all
@test objective_value_all ≈ objective_value_lazy
@test objective_value_all ≈ minimum_value
@test reduce(&, [minimum[i] > solution_all[i][1] && minimum[i] < solution_all[i][2] for i in 1:3])
