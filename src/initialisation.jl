current_dir =  @__DIR__
cd(current_dir)

using Pkg
Pkg.instantiate()

using MLDatasets, CUDA, FileIO, ImageShow 
using MLJBase # for conf matrix
using Plots, Images
using Statistics
using Random
using Serialization
using Flux
using Flux: params, train!, mse, flatten, onehotbatch
using JuMP
using JuMP: Model, value
using HiGHS
using Gurobi
using EvoTrees
using CSV
using DataFrames
using StatsBase
using MLJ
using JLD

include("JuMP_model.jl")
include("MNIST.jl")
include("neural_nets.jl")
include("bound_tightening.jl")

include.(filter(contains(r".jl$"), readdir(current_dir*"/decision_trees/"; join=true)))




