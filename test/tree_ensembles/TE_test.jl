using EvoTrees
using JuMP
using Gurobi

evo_model = EvoTrees.load("tree_ensembles/paraboloid.bson");

universal_tree_model = extract_evotrees_info(evo_model)

solution_lazy, objective_value_lazy, jump_model_lazy = tree_model_to_MIP(universal_tree_model; objective=MIN_SENSE, create_initial = false)
solution, objective_value, jump_model = tree_model_to_MIP(universal_tree_model; objective=MIN_SENSE, create_initial = true)

minimum_value = -0.0008044997f0;
minimum = [ -0.0373748243818212, -0.042113434576107486,  0.009143172249250781];

@test solution_lazy ≈ solution
@test objective_value ≈ objective_value_lazy
@test objective_value ≈ minimum_value
@test reduce(&, [minimum[i] > solution[i][1] && minimum[i] < solution[i][2] for i in 1:3])
