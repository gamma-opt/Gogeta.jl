module Gogeta

#include the files for neural networks functions and export the functions

include("neural_networks/JuMP_model.jl")
export create_JuMP_model, evaluate!

include("neural_networks/CNN_JuMP_model.jl")
export create_CNN_JuMP_model, evaluate_CNN!

include("neural_networks/bound_tightening.jl")
export bound_tightening,
    bound_tightening_threads,
    bound_tightening_workers,
    bound_tightening_2workers

#include the files for tree tree_ensembles functions and export the functions
include("tree_ensembles/util.jl")

include("tree_ensembles/tree_model_to_MIP.jl")
export tree_model_to_MIP

include("tree_ensembles/types.jl")
export TEModel,
    extract_evotrees_info

end