# Create a random sample in the domain x1, x2 in (-0.5, 0.5) of the function f(x1, x2) = x1^2 + x2^2 (circular paraboloid)

data = rand(1000, 2) .- 0.5;

x_train = data[1:750, :];
y_train = vec(sum(map.(x->x^2, x_train), dims=2));

x_test = data[751:end, :];
y_test = vec(sum(map.(x->x^2, x_test), dims=2));

# Train an EvoTrees (gradient-boosted trees) model with the sample

using EvoTrees

config = EvoTreeRegressor(nrounds=500, max_depth=5);
evo_model = fit_evotree(config; x_train, y_train);

# Use model to make predictions

using Statistics

pred_train = EvoTrees.predict(evo_model, x_train);
pred_test = EvoTrees.predict(evo_model, x_test);
r2_score_train = 1 - sum((y_train .- pred_train).^2) / sum((y_train .- mean(y_train)).^2)
r2_score_test = 1 - sum((y_test .- pred_test).^2) / sum((y_test .- mean(y_test)).^2)

using JuMP
using Gurobi
using Gogeta

# Extract data from EvoTrees model

universal_tree_model = extract_evotrees_info(evo_model);

# Create JuMP model and solve the tree ensemble optimization problem (input that minimizes output)

solution, objective_value, jump_model = tree_model_to_MIP(universal_tree_model; objective=MIN_SENSE, create_initial = false);
solution_init, objective_value_init, jump_model_init = tree_model_to_MIP(universal_tree_model; objective=MIN_SENSE, create_initial = true);