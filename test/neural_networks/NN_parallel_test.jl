using Distributed

addprocs(4)
@everywhere using Gogeta

solver_params = SolverParams(solver="HiGHS", silent=true, threads=0, relax=false, time_limit=1.0);

@info "Creating a JuMP model from the neural network with bound tightening."
@time nn_parallel, U_parallel, L_parallel = NN_to_MIP_with_bound_tightening(model, init_U, init_L, solver_params; bound_tightening="standard");

@info "Testing that correct model is produced"
@test vec(model(x)) ≈ [forward_pass!(nn_parallel, input)[] for input in eachcol(x)]

@info "Testing that parallel computed bounds are the same."
@test U_parallel ≈ U_correct
@test L_parallel ≈ L_correct

rmprocs(workers())