module Gogeta

include("neural_networks/JuMP_model.jl")
export create_JuMP_model, evaluate!

include("neural_networks/CNN_JuMP_model.jl")
export create_CNN_JuMP_model, evaluate_CNN!

include("neural_networks/bound_tightening.jl")
export bound_tightening,
    bound_tightening_threads,
    bound_tightening_workers,
    bound_tightening_2workers

include("tree_ensembles/types.jl")
export TEModel, extract_evotrees_info

include("tree_ensembles/util.jl")
export get_solution

include("tree_ensembles/TE_to_MIP.jl")
export TE_to_MIP, optimize_with_initial_constraints!, optimize_with_lazy_constraints!

end
