module ML_as_MO

include("JuMP_model.jl")
include("bound_tightening.jl")

export create_JuMP_model,
    evaluate!

export bound_tightening,
    bound_tightening_threads
    bound_tightening_workers
    bound_calculating

end # module