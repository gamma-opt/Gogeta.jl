using Flux
using Random
using Gogeta
using Gurobi
using JuMP
using QuasiMonteCarlo

dimension = 10

begin
    Random.seed!(1234);

    NN_model = Chain(
        Dense(dimension => 100, relu),
        Dense(100 => 1)
    )
end

# Set upper and lower input bounds
init_U = ones(dimension);
init_L = zeros(dimension);

# Formulate the MIP with heuristic bound tightening
jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model)
NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="fast");

last_layer, _ = maximum(keys(jump_model[:x].data))
@objective(jump_model, Max, jump_model[:x][last_layer, 1])

samples = QuasiMonteCarlo.sample(1000, init_L, init_U, LatinHypercubeSample());
@time x_opt, optimum = optimize_by_sampling!(jump_model, samples);
optimum
NN_model(Float32.(x_opt))[] ≈ optimum

@time optimize!(jump_model)
objective_value(jump_model)