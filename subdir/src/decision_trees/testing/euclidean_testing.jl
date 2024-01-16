using EvoTrees
using Plots
using Random
using Statistics
using Interpolations
using JuMP
using Gurobi
include("tree_model_to_MIP.jl")
include("util.jl")
include("types.jl")

const ENV = Gurobi.Env()

"DATA GENERATION"

Random.seed!(1)
n_feats = 5
data = randn(1000, n_feats);
data = data[shuffle(1:end), :];

split::Int = floor(0.75 * length(data[:, 1]))

x_train = data[1:split, :];
y_train = Array{Float64}(undef, length(x_train[:, 1]));
[y_train[i] = sqrt(sum(x_train[i, :].^2)) for i in 1:length(y_train)];

x_test = data[split+1:end, :];
y_test = Array{Float64}(undef, length(x_test[:, 1]));
[y_test[i] = sqrt(sum(x_test[i, :].^2)) for i in 1:length(y_test)];

"TREE MODEL CONFIGURATION AND TRAINING"

tree_depth, forest_size = 5, 500

config = EvoTreeRegressor(nrounds=forest_size, max_depth=tree_depth, T=Float64, loss=:linear);
model = fit_evotree(config; x_train, y_train);

pred_train = EvoTrees.predict(model, x_train);
pred_test = EvoTrees.predict(model, x_test);

"OPTIMIZATION"

universal_model = extract_evotrees_info(model);
@time x_new, m_new = tree_model_to_MIP(universal_model; create_initial=true, objective=MIN_SENSE, gurobi_env=ENV);
@time x_alg, m_alg = tree_model_to_MIP(universal_model; create_initial=false, objective=MIN_SENSE, gurobi_env=ENV);

"CHECKING OF SOLUTION"

objective_value(m_new)
EvoTrees.predict(model, reshape([mean(x_new[n]) for n in 1:n_feats], 1, n_feats))[1]

objective_value(m_alg)
EvoTrees.predict(model, reshape([mean(x_alg[n]) for n in 1:n_feats], 1, n_feats))[1]

"METRICS"

println("N_levels: $(length(eachindex(m_new[:x])))")
println("N_leaves: $(length(eachindex(m_new[:y])))")
println("Maxium possible number of leaves: $(2^(tree_depth-1)*forest_size)")