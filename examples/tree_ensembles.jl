# Create a random sample in the domain x1, x2 in (-0.5, 0.5) of the function f(x1, x2) = x1^2 + x2^2 (circular paraboloid)
data = rand(1000, 2) .- 0.5;

x_train = data[1:750, :];
y_train = vec(sum(map.(x->x^2, x_train), dims=2));

x_test = data[751:end, :];
y_test = vec(sum(map.(x->x^2, x_test), dims=2));

# Train an EvoTrees (gradient-boosted trees) model with the sample

using EvoTrees

config = EvoTreeRegressor(nrounds=500, max_depth=5);
evo_model = fit_evotree(config; x_train, y_train, verbosity=0);

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
const ENV = Gurobi.Env();
jump = Model(() -> Gurobi.Optimizer(ENV));
set_attribute(jump, "OutputFlag", 0) # JuMP or solver-specific attributes can be changed

TE_formulate!(jump, universal_tree_model, MIN_SENSE);

# Solve first by creating all split constraints
add_split_constraints!(jump, universal_tree_model)
optimize!(jump)

# Show results
get_solution(jump, universal_tree_model)
objective_value(jump)

# Then solve with lazy constaints
# For lazy callback, model must be direct
jump = direct_model(Gurobi.Optimizer(ENV));
set_silent(jump)

TE_formulate!(jump, universal_tree_model, MIN_SENSE);

# Define callback function. For each solver this might be slightly different.
# Inside the callback 'tree_callback_algorithm' must be called.

function split_constraint_callback_gurobi(cb_data, cb_where::Cint)

    # Only run at integer solutions
    if cb_where != GRB_CB_MIPSOL
        return
    end

    Gurobi.load_callback_variable_primal(cb_data, cb_where)
    tree_callback_algorithm(cb_data, universal_tree_model, jump)

end

set_attribute(jump, "LazyConstraints", 1)
set_attribute(jump, Gurobi.CallbackFunction(), split_constraint_callback_gurobi)

optimize!(jump)

# Show results
get_solution(jump, universal_tree_model)
objective_value(jump)
