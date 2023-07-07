using EvoTrees
using Plots
using Random
using Statistics
using Interpolations
using JuMP
using Gurobi
using CSV
using DataFrames

include("tree_model_to_MIP.jl");
include("util.jl");
include("types.jl");

const ENV = Gurobi.Env();

"DATA LOADING"

dataset_name = "OX2";
train_data, x_train, y_train, x_test, y_test, feat_names = load_drug_data(dataset_name);

"TREE MODEL GENERATION"

trees = [10, 50, 100, 250, 500];
depth = 10;

config = EvoTreeRegressor(nrounds=maximum(trees), max_depth=depth);
train_time = @elapsed model = fit_evotree(config, train_data[:, 2:end]; target_name="Act", verbosity=0, fnames=feat_names);
result_file = open(string(@__DIR__)*"/drug_test_results.txt", "a");
write(result_file, "\nDataset: $dataset_name, Trees: $(maximum(trees)), Depth: $depth, Train time: $(train_time)\n");
close(result_file)

EvoTrees.save(model, string(@__DIR__)*"/trained_models/$(dataset_name)_$(maximum(trees))_trees_$(depth)_depth.bson")

loaded_model = EvoTrees.load(string(@__DIR__)*"/trained_models/$(dataset_name)_$(maximum(trees))_trees_$(depth)_depth.bson");

for forest_size in trees

    pred_train = EvoTrees.predict(loaded_model, x_train; ntree_limit=forest_size);
    pred_test = EvoTrees.predict(loaded_model, x_test; ntree_limit=forest_size);

    r2_score_train = 1 - sum((y_train .- pred_train).^2) / sum((y_train .- mean(y_train)).^2)
    r2_score_test = 1 - sum((y_test .- pred_test).^2) / sum((y_test .- mean(y_test)).^2)

    result_file = open(string(@__DIR__)*"/drug_test_results.txt", "a");
    write(result_file, "Dataset: $dataset_name, Trees: $forest_size, Depth: $depth, R2 train: $(r2_score_train), R2 test: $(r2_score_test)\n");
    close(result_file)

end

"OPTIMIZATION"

loaded_model = EvoTrees.load(string(@__DIR__)*"/trained_models/OX2_500_trees_3_depth.bson");
universal_model = extract_evotrees_info(loaded_model);
@time x_new, m_new = tree_model_to_MIP(universal_model; create_initial=true, objective=MAX_SENSE, gurobi_env=ENV);
@time x_alg, m_alg = tree_model_to_MIP(universal_model; create_initial=false, objective=MAX_SENSE, gurobi_env=ENV);

"CHECKING OF SOLUTION"

objective_value(m_new)
EvoTrees.predict(loaded_model, reshape([mean(x_new[n]) for n in eachindex(x_new)], 1, length(x_new)))[1]

objective_value(m_alg)
EvoTrees.predict(loaded_model, reshape([mean(x_alg[n]) for n in eachindex(x_alg)], 1, length(x_alg)))[1]

"METRICS"

println("N_levels: $(length(eachindex(m_new[:x])))")
println("N_leaves: $(length(eachindex(m_new[:y])))")