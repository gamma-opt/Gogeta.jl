using EvoTrees
using JuMP
using GLPK
using Gurobi

include("TE_to_MIP.jl");
include("types.jl");
include("util.jl");

evo_model = EvoTrees.load(string(@__DIR__)*"/../../test/tree_ensembles/paraboloid.bson");
universal_tree_model = extract_evotrees_info(evo_model);

TE_to_MIP(universal_tree_model, GLPK.Optimizer(); objective=MIN_SENSE, create_initial = true, timelimit=10)
TE_to_MIP(universal_tree_model, GLPK.Optimizer(); objective=MIN_SENSE, create_initial = false, timelimit=10)

TE_to_MIP(universal_tree_model, Gurobi.Optimizer(); objective=MIN_SENSE, create_initial = true, timelimit=10)
TE_to_MIP(universal_tree_model, Gurobi.Optimizer(); objective=MIN_SENSE, create_initial = false, timelimit=10)
