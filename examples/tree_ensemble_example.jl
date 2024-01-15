data = rand(1000, 2) .- 0.5;

x_train = data[1:750, :];
y_train = vec(sum(map.(x->x^2, x_train), dims=2));

x_test = data[751:end, :];
y_test = vec(sum(map.(x->x^2, x_test), dims=2));

using EvoTrees

config = EvoTreeRegressor(nrounds=500, max_depth=5);
evo_model = fit_evotree(config; x_train, y_train);

pred_train = EvoTrees.predict(evo_model, x_train);
pred_test = EvoTrees.predict(evo_model, x_test);

using JuMP
using Gurobi
using Gogeta

universal_tree_model = extract_evotrees_info(evo_model);

solution, objective_value, jump_model = tree_model_to_MIP(universal_tree_model; objective=MIN_SENSE, create_initial = false);
solution_init, objective_value_init, jump_model_init = tree_model_to_MIP(universal_tree_model; objective=MIN_SENSE, create_initial = true);

println("Predicition of EvoTrees model given the otimal solution of correspoding MIP problem as an input")
println("Prediction of EvoTrees model for the solution of MIP problem solved without lazy constraints algorithm: $(EvoTrees.predict(model, reshape([mean(x_new[n]) for n in 1:8], 1, 8))[1])")
println("Prediction of EvoTrees model for the solution of MIP problem solved with lazy constraints algorithm: $(EvoTrees.predict(model, reshape([mean(x_alg[n]) for n in 1:8], 1, 8))[1])")
println("Maximum conceivable sum of tree predictions: $(sum(maximum(model.trees[tree].pred) for tree in 1:1001))")
println("Optimal objective value of the MIP problem solved without lazy constraints algorithm: $sol_new")
println("Optimal objective value of the MIP problem solved with lazy constraints algorithm: $sol_alg")
println("Maximum of the test dataset: $(maximum(pred_test))")