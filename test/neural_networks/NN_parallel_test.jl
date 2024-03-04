using Distributed

addprocs(4)
@everywhere using HiGHS
@everywhere using JuMP

@everywhere function set_solver!(jump)
    set_optimizer(jump, () -> HiGHS.Optimizer())
    set_silent(jump)
    set_attribute(jump, "time_limit", 1.0)
end

@everywhere using Gogeta

@info "Creating a JuMP model from the neural network with bound tightening."
jump = Model()
set_solver!(jump)
U_parallel, L_parallel = NN_formulate!(jump, NN_model, init_U, init_L; bound_tightening="standard", silent=false, parallel=true);

@info "Testing that correct model is produced"
@test vec(NN_model(x)) ≈ [forward_pass!(jump, input)[] for input in eachcol(x)]

@info "Testing that parallel computed bounds are the same."
@test U_parallel ≈ U_correct
@test L_parallel ≈ L_correct

rmprocs(workers())