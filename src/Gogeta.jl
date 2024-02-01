module Gogeta

include("neural_networks/NN_to_MIP.jl")
export NN_to_MIP, forward_pass!, SolverParams

include("neural_networks/bounds.jl")

include("tree_ensembles/types.jl")
export TEModel, extract_evotrees_info

include("tree_ensembles/util.jl")
export get_solution

include("tree_ensembles/TE_to_MIP.jl")
export TE_to_MIP, optimize_with_initial_constraints!, optimize_with_lazy_constraints!

end