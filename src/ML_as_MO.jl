module ML_as_MO

using MLDatasets, FileIO, ImageShow 
using MLJBase # for conf matrix
using Statistics
using Random
using Flux
using Flux: params, train!, mse, flatten, onehotbatch
using JuMP
using JuMP: Model, value
using Gurobi
using EvoTrees
using CSV
using DataFrames
using StatsBase
using MLJ
using JLD
using Profile
#using Plots
using Interpolations
using XLSX
using Distributed, SharedArrays

include("nn/JuMP_model.jl")
include("nn/bound_tightening.jl")
include("nn/CNN_JuMP_model.jl")


include("decision_trees/types.jl")
include("decision_trees/util.jl")
include("decision_trees/tree_model_to_MIP.jl")

# read all the files in decision_trees->visualisation
#include("decision_trees/visualization/opt_visualization.jl")
#include("decision_trees/visualization/plotting.jl")
#include("decision_trees/visualization/quality_visualization.jl")

export create_JuMP_model,
    evaluate!

export bound_tightening,
    bound_tightening_threads,
    bound_tightening_workers,
    bound_tightening_2workers

export create_CNN_JuMP_model,
    evaluate_CNN!

export trees_to_relaxed_MIP,
    extract_tree_model_info,
    children

end # module