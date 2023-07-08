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
include("testing_functions.jl");
include("types.jl");

const ENV = Gurobi.Env();

"DATA LOADING"

dataset_name = "OX2";
train_data, x_train, y_train, x_test, y_test, feat_names = load_drug_data(dataset_name);

"TREE MODEL GENERATION"

trees = [10, 50, 100, 200, 350, 500, 750, 1000];
depths = [3, 5, 7, 10, 12];

train_evo_models(depths, trees, train_data, feat_names, x_train, y_train, x_test, y_test)

"OPTIMIZATION"

for forest_size in trees, depth in depths

    loaded_model = EvoTrees.load(string(@__DIR__)*"/trained_models/$(dataset_name)_$(forest_size)_trees_$(depth)_depth.bson");
    universal_model = extract_evotrees_info(loaded_model);

    time_normal = @elapsed x_new, m_new = tree_model_to_MIP(universal_model; create_initial=true, objective=MAX_SENSE, gurobi_env=ENV);
    time_alg = @elapsed x_alg, m_alg = tree_model_to_MIP(universal_model; create_initial=false, objective=MAX_SENSE, gurobi_env=ENV);
    
    println("N_levels: $(length(eachindex(m_new[:x])))")
    println("N_leaves: $(length(eachindex(m_new[:y])))")

    result_file = open(string(@__DIR__)*"/drug_opt_results.txt", "a");
    write(result_file, "Dataset: $dataset_name, 
                        Trees: $(maximum(trees)), 
                        Depth: $depth, 
                        Normal time: $(time_normal), 
                        Alg time: $(time_alg)
                        N levels: $(length(eachindex(m_new[:x])))
                        N leaves: $(length(eachindex(m_new[:y])))\n");
    close(result_file)
end
