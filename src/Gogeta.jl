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

# Write your package code here.


end
