module Gogeta

#include the files for neural networks functions and export the functions

include("nn/JuMP_model.jl")
export create_JuMP_model, evaluate!

include("nn/CNN_JuMP_model.jl")
export create_CNN_JuMP_model, evaluate_CNN!

include("nn/bound_tightening.jl")
export bound_tightening,
    bound_tightening_threads,
    bound_tightening_workers,
    bound_tightening_2workers

#include the files for decision trees functions and export the functions

include("dt/tree_model_to_MIP.jl")
export tree_model_to_MIP

include("dt/types.jl")
export TEModel

include("dt/util.jl")
export extract_evotrees_info,
        children

# Write your package code here.


end
