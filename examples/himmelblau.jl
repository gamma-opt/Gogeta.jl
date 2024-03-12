using Flux
using Random
using Gogeta
using Gurobi
using JuMP
using Plots
using Revise
using QuasiMonteCarlo

begin
    Random.seed!(12345);

    NN_model = Chain(
        Dense(2 => 100, relu),
        Dense(100 => 100, relu),
        Dense(100 => 1)
    )
end

# Set upper and lower input bounds
init_U = [5.0, 5.0];
init_L = [-5.0, -5.0];

x_range = LinRange{Float32}(init_L[1], init_U[1], 50);
y_range = LinRange{Float32}(init_L[2], init_U[2], 50);

f_himmelblau(x, y) = ((x^2+y-11)^2 + (x+y^2-7)^2)
f(x, y) = -log(f_himmelblau(x, y))

plot(x_range, y_range, f, st=:surface, c=cgrad(:viridis), camera=(-40, 50))
plot(x_range, y_range, f, st=:contourf, c=cgrad(:viridis), lw=0)
plot(x_range, y_range, (x, y) -> NN_model([x y]')[], st=:contourf, c=cgrad(:viridis), lw=0)

opt_setup = Flux.setup(Adam(), NN_model)
samples = QuasiMonteCarlo.sample(10_000, init_L, init_U, LatinHypercubeSample());
train_set = map(p -> (p, f(p[1], p[2])), eachcol(samples))
loss(y_hat, y) = sum((y_hat .- y).^2)

plot(x_range, y_range, (x, y) -> NN_model([x y]')[], st=:surface, c=cgrad(:viridis), camera=(-40, 50))
for _ in 1:10
    Flux.train!(NN_model, train_set, opt_setup) do m,x,y
        loss(m(x), y)
    end
    display(plot(x_range, y_range, (x, y) -> NN_model([x y]')[], st=:surface, c=cgrad(:viridis), camera=(-40, 50)))
end

plot(x_range, y_range, f, st=:surface, c=cgrad(:viridis), camera=(-40, 50))
plot(x_range, y_range, (x, y) -> NN_model([x y]')[], st=:surface, c=cgrad(:viridis), camera=(-40, 50))
plot(x_range, y_range, (x, y) -> NN_model([x y]')[], st=:contourf, c=cgrad(:viridis), lw=0)

using BSON: @save, @load
# @save "himmelblau_model.bson" NN_model

@load "examples/himmelblau_model.bson" NN_model

# Formulate the MIP with heuristic bound tightening
jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model)
NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="fast");

last_layer, _ = maximum(keys(jump_model[:x].data))
@objective(jump_model, Max, jump_model[:x][last_layer, 1])

include("../src/neural_networks/relaxing_walk.jl")
@time opt, samples = optimize_by_walking!(jump_model, init_U, init_L; return_sampled=true, iterations=5, samples_per_iter=3);

plot(x_range, y_range, (x, y) -> NN_model([x y]')[], st=:contourf, c=cgrad(:viridis), lw=0)
# savefig("himmelblau.png")
scatter!(first.(samples), last.(samples), label="samples", color=:blue)
scatter!(first.(unique(opt)), last.(unique(opt)), label="optima", color=:red)
[display(plot!([samples[i][1], opt[i][1]], [samples[i][2], opt[i][2]], lw=1, color=:black, label=false)) for i in eachindex(samples)];

copy_model = copy(jump_model)
set_solver!(copy_model)
relax_integrality(copy_model)
@time plot(x_range, y_range, (x, y) -> forward_pass!(copy_model, [x, y])[], st=:contourf, c=cgrad(:viridis), lw=0)

function set_solver!(jump)
    set_optimizer(jump, Gurobi.Optimizer)
    set_silent(jump)
end