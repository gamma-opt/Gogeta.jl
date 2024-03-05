using Flux
using Random
using Gogeta
using Gurobi
using JuMP
using Revise
using QuasiMonteCarlo

begin
    Random.seed!(1234);

    NN_model = Chain(
        Dense(10 => 100, relu),
        Dense(100 => 1)
    )
end

# Set upper and lower input bounds
init_U = ones(10)
init_L = zeros(10)

# Formulate the MIP with optimization-based bound tightening
jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model)
bounds_U, bounds_L = NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="fast");

last_layer, _ = maximum(keys(jump_model[:x].data))
@objective(jump_model, Max, jump_model[:x][last_layer, 1])

@time optimize!(jump_model)
objective_value(jump_model)

samples = QuasiMonteCarlo.sample(1000, init_L, init_U, LatinHypercubeSample());
@time x_opt, optimum = optimize_by_sampling!(jump_model, samples);
optimum
NN_model(x_opt)[] ≈ optimum