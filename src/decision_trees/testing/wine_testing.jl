using EvoTrees
using Plots
using Random
using Statistics
using Interpolations
using JuMP
using Gurobi
using CSV
using DataFrames
include("tree_model_to_MIP.jl")
include("util.jl")
include("types.jl")

const ENV = Gurobi.Env()

"DATA GENERATION"

cd(@__DIR__)
data = Matrix(CSV.read("data/winequality-red.csv", DataFrame));

Random.seed!(1)
data = data[shuffle(1:end), :]

train_split::Int = floor(0.75 * length(data[:, 1]))

x_train = data[1:train_split, 1:11];
y_train = data[1:train_split, 12];

x_test = data[train_split+1:end, 1:11];
y_test = data[train_split+1:end, 12];

n_feats = 11;

"TREE MODEL CONFIGURATION AND TRAINING"

forest_size = 250;
tree_depth = 5;

config = EvoTreeRegressor(nrounds=forest_size, max_depth=tree_depth);
model = fit_evotree(config; x_train, y_train, verbosity=0);

pred_train = EvoTrees.predict(model, x_train);
pred_test = EvoTrees.predict(model, x_test);

r2_score_test = 1 - sum((y_test .- pred_test).^2) / sum((y_test .- mean(y_test)).^2)

"OPTIMIZATION"

universal_model = extract_evotrees_info(model);
@time x_new, m_new = tree_model_to_MIP(universal_model; create_initial=true, objective=MAX_SENSE, gurobi_env=ENV);
@time x_alg, m_alg = tree_model_to_MIP(universal_model; create_initial=false, objective=MAX_SENSE, gurobi_env=ENV);

"CHECKING OF SOLUTION"

objective_value(m_new)
EvoTrees.predict(model, reshape([mean(x_new[n]) for n in 1:n_feats], 1, n_feats))[1]

objective_value(m_alg)
EvoTrees.predict(model, reshape([mean(x_alg[n]) for n in 1:n_feats], 1, n_feats))[1]

"METRICS"

println("N_levels: $(length(eachindex(m_new[:x])))")
println("N_leaves: $(length(eachindex(m_new[:y])))")
println("Maxium possible number of leaves: $(2^(tree_depth-1)*forest_size)")