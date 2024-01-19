using EvoTrees
using JuMP
using SCIP
using GLPK
using Gurobi
using Gogeta

evo_model = EvoTrees.load(string(@__DIR__)*"/../../test/tree_ensembles/paraboloid.bson");
universal_tree_model = extract_evotrees_info(evo_model);

const ENV = Gurobi.Env();
model = TE_to_MIP(universal_tree_model, Gurobi.Optimizer(ENV), MIN_SENSE)
set_attribute(model, "OutputFlag", 0)

@time optimize_with_initial_constraints!(model, universal_tree_model)
@time optimize_with_lazy_constraints!(model, universal_tree_model)

model = TE_to_MIP(universal_tree_model, GLPK.Optimizer(), MIN_SENSE)
@time optimize_with_lazy_constraints!(model, universal_tree_model)

model = TE_to_MIP(universal_tree_model, SCIP.Optimizer(), MIN_SENSE)
@time optimize_with_initial_constraints!(model, universal_tree_model)

get_solution(model, universal_tree_model)
objective_value(model)