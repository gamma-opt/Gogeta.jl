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
using Interpolations
using XLSX
using Distributed, SharedArrays

include("nn/JuMP_model.jl")
include("nn/bound_tightening.jl")

export create_JuMP_model,
    evaluate!

export bound_tightening,
    bound_tightening_threads,
    bound_tightening_workers
    # bound_calculating # inner function does not need to be public

end # module