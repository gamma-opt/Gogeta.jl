using Flux
using Random
using Gogeta
using GLPK
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

@info "Formulating the MIP with heuristic bound tightening."
jump_model = Model(GLPK.Optimizer)
set_silent(jump_model)
NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="fast");

last_layer, _ = maximum(keys(jump_model[:x].data))
@objective(jump_model, Min, jump_model[:x][last_layer, 1])

@info "Solving by sampling."

Random.seed!(1234);
samples = QuasiMonteCarlo.sample(5, init_L, init_U, LatinHypercubeSample())
x_opt, extremum = optimize_by_sampling!(jump_model, samples);

@info "Testing that sampling solution and extremum agree."
@test NN_model(Float32.(x_opt))[] ≈ extremum

@info "Testing that optimum is correct (precomputed)."
@test x_opt ≈ [1.0, 0.731267784484517, 0.0, 0.6963412902527348, 0.9324087029309082, 0.12149594815045718, 0.0, 1.0, 1.0, 1.0]
@test extremum ≈ -0.7496729517399346