using EvoTrees
using Plots
using Random
using Statistics
using Interpolations
using JuMP
using Gurobi
include("trees_to_relaxed_op.jl")
include("util.jl")

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

"CONCRETE DATA IMPORT"

using XLSX

cd(@__DIR__)
xf = XLSX.readxlsx("data/Concrete_Data.xlsx");
data = Float64.(xf["Sheet1"]["A2:I1031"]);

Random.seed!(1)
data = data[shuffle(1:end), :]

split::Int = floor(0.75 * length(data[:, 1]))

x_train = data[1:split, 1:8];
y_train = data[1:split, 9];

x_test = data[split+1:end, 1:8];
y_test = data[split+1:end, 9];

n_feats = 8

"TREE MODEL CONFIGURATION AND TRAINING"

tree_depth, forest_size = 15, 50

config = EvoTreeRegressor(nrounds=forest_size, max_depth=tree_depth, T=Float64, loss=:linear);
model = fit_evotree(config; x_train, y_train);

pred_train = EvoTrees.predict(model, x_train);
pred_test = EvoTrees.predict(model, x_test);

"OPTIMIZATION"

@time x_new, sol_new, m_new = trees_to_relaxed_MIP(model, tree_depth; constraints="initial", objective="max");
@time x_alg, sol_alg, m_alg = trees_to_relaxed_MIP(model, tree_depth; constraints="generate", objective="max");

"CHECKING OF SOLUTION"

sol_new
EvoTrees.predict(model, reshape([mean(x_new[n]) for n in 1:n_feats], 1, n_feats))[1]

sol_alg
EvoTrees.predict(model, reshape([mean(x_alg[n]) for n in 1:n_feats], 1, n_feats))[1]

"METRICS"

println("N_levels: $(length(eachindex(m_new[:x])))")
println("N_leaves: $(length(eachindex(m_new[:y])))")
println("Maxium possible number of leaves: $(2^(tree_depth-1)*forest_size)")