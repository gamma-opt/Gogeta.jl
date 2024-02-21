using Flux
using Random

# Create a small neural network with random weights
begin
    Random.seed!(1234);

    model = Chain(
        Dense(2 => 10, relu),
        Dense(10 => 50, relu),
        Dense(50 => 20, relu),
        Dense(20 => 5, relu),
        Dense(5 => 1)
    )
end

# Set upper and lower input bounds
init_U = [-0.5, 0.5];
init_L = [-1.5, -0.5];

using Distributed
addprocs(4)
@everywhere using Gogeta

solver_params = SolverParams(solver="Gurobi", silent=true, threads=0, relax=false, time_limit=0);

# Create a JuMP model from the neural network with bound tightening.
@time nn_parallel, U_parallel, L_parallel = NN_to_MIP_with_bound_tightening(model, init_U, init_L, solver_params; bound_tightening="standard");

# Bound tightening is 'automatically' run in parallel if multiple workers are recognized

rmprocs(workers())