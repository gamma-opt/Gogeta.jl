module Gogeta

using Flux
using JuMP
using LinearAlgebra: rank, dot
using EvoTrees
using Distributed

# Neural networks

include("neural_networks/bounds.jl")
include("neural_networks/compression.jl")
export NN_compress

include("neural_networks/NN_to_MIP.jl")
export NN_formulate!
export forward_pass!

include("neural_networks/CNN_util.jl")
export CNNStructure, get_structure, image_pass!

include("neural_networks/CNN_to_MIP.jl")
export CNN_formulate!

# Sampling
include("neural_networks/sampling.jl")
export optimize_by_sampling!

# Tree ensembles

include("tree_ensembles/types.jl")
export TEModel, extract_evotrees_info

include("tree_ensembles/util.jl")
export get_solution

include("tree_ensembles/TE_to_MIP.jl")
export TE_formulate!
export add_split_constraints!, tree_callback_algorithm

end