using Flux
using Random
using Gogeta
using Gurobi
using JuMP
using Plots
using Revise

dimension = 2

begin
    Random.seed!(12345);

    NN_model = Chain(
        Dense(dimension => 100, relu),
        Dense(100 => 100, relu),
        Dense(100 => 1)
    )
end

# Set upper and lower input bounds
init_U = ones(dimension);
init_L = zeros(dimension);

init_U = [5.0, 5.0];
init_L = [-5.0, -5.0];

# Formulate the MIP with heuristic bound tightening
jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model)
NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="fast");

last_layer, _ = maximum(keys(jump_model[:x].data))
@objective(jump_model, Max, jump_model[:x][last_layer, 1])

x_range = LinRange{Float32}(init_L[1], init_U[1], 50);
y_range = LinRange{Float32}(init_L[2], init_U[2], 50);

f_himmelblau(x, y) = ((x^2+y-11)^2 + (x+y^2-7)^2)
f(x, y) = -log(f_himmelblau(x, y))

plot(x_range, y_range, f_himmelblau, st=:surface, c=cgrad(:viridis), camera=(-40, 50))
plot(x_range, y_range, f, st=:contourf, c=cgrad(:viridis), lw=0)

plot(x_range, y_range, (x, y) -> NN_model(hcat(x, y)')[], st=:surface, c=cgrad(:viridis), camera=(-40, 50))

plot(x_range, y_range, (x, y) -> NN_model(hcat(x, y)')[], st=:contourf, c=cgrad(:viridis), camera=(-40, 50))

relax_integrality(jump_model)
plot(x_range, y_range, (x, y) -> forward_pass!(jump_model, [x,y])[], st=:contourf, c=cgrad(:viridis), camera=(-40, 50))

for x in range(-2.9, 2.9, 10), y in range(-2.9, 2.9, 10)
    _, path = local_search([x, y], jump_model, init_U, init_L; show_path=true)
    heights = [NN_model(path[i])[] for i in eachindex(path)]
    display(plot!(first.(path), last.(path), heights .+ 0.1, marker=:circle, legend=false, lw=3))
    sleep(0.1)
end

_, path = local_search([2.0, 2.0], jump_model, init_U, init_L; show_path=true)
heights = [NN_model(path[i])[] for i in eachindex(path)]
display(plot!(first.(path), last.(path), heights, marker=:circle, legend=false, lw=5))

opt, samples = optimize_by_walking!(jump_model, init_U, init_L; return_sampled=true);
solutions = filter(v -> !iszero(v), results)
scatter!(first.(solutions), last.(solutions), legend=false)

contourf(x_range, y_range, (x, y) -> NN_model(hcat(x, y)')[], c=cgrad(:viridis), lw=0)
using QuasiMonteCarlo
samples = QuasiMonteCarlo.sample(10, init_L, init_U, LatinHypercubeSample());
for sample in eachcol(samples)
    display(scatter!([sample[1]], [sample[2]], legend=false, color=:blue))
    res = local_search(sample, jump_model, init_U, init_L)
    display(scatter!([res[1]], [res[2]], legend=false, color=:red))
   #display(plot!(sample, res, color=:black, linewidth=2, legend=false))
end

function set_solver!(jump)
    set_optimizer(jump, Gurobi.Optimizer)
    set_silent(jump)
end