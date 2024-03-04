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

using Distributed
addprocs(2)
@everywhere using Gurobi

@everywhere ENV = Ref{Gurobi.Env}()

@everywhere function init_env()
    global ENV
    ENV[] = Gurobi.Env()
end

fetch(@spawnat 2 init_env())
fetch(@spawnat 3 init_env())

@everywhere using JuMP

@everywhere function set_solver!(jump)
    set_optimizer(jump, () -> Gurobi.Optimizer(ENV[]))
    set_silent(jump)
    return jump
end

# Set upper and lower input bounds
init_U = [1.0, 1.0];
init_L = [-1.0, -1.0];

@everywhere using Gogeta

# Create a JuMP model from the neural network with parallel bound tightening.
jump = Model()
U, L = NN_formulate!(jump, model, init_U, init_L; bound_tightening="standard", silent=false, parallel=true);