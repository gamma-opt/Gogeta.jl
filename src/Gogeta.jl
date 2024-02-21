module Gogeta

using Flux
using JuMP
using LinearAlgebra: rank, dot
using Gurobi
using GLPK
using HiGHS
using Distributed
using EvoTrees

const GUROBI_ENV = Ref{Gurobi.Env}()

function __init__()
    try
        const GUROBI_ENV[] = Gurobi.Env()
    catch e
        println(e)
        @warn "Gurobi is not able to be used."
    end
end

include("neural_networks/bounds.jl")
include("neural_networks/prune.jl")
include("neural_networks/NN_to_MIP.jl")

include("neural_networks/interface.jl")
export NN_to_MIP_with_precomputed, NN_to_MIP_with_bound_tightening, compress_with_precomputed, compress_with_bound_tightening
export forward_pass!, SolverParams

include("tree_ensembles/types.jl")
export TEModel, extract_evotrees_info

include("tree_ensembles/util.jl")
export get_solution

include("tree_ensembles/TE_to_MIP.jl")
export TE_to_MIP, optimize_with_initial_constraints!, optimize_with_lazy_constraints!

end