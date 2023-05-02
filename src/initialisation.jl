current_dir =  @__DIR__
cd(current_dir)

using Pkg
Pkg.instantiate()

using MLDatasets, CUDA, FileIO, ImageShow 
using MLJBase # for conf matrix
using Plots, Images
using Statistics
using Random
using ImageBinarization
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

include("JuMP_model.jl")
include("file_read_write.jl")
include("MNIST.jl")
include("neural_nets.jl")




