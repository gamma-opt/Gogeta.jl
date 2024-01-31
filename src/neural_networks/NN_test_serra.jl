data = rand(Float32, (2, 1000)) .- 0.5f0;
x_train = data[:, 1:750];
y_train = [sum(x_train[:, col].^2) for col in 1:750];

using Distributed
using Flux
using Random
using Revise
using Gogeta

addprocs(7)

@everywhere using Gogeta

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

begin
    Random.seed!(1234);

    model = Chain(
        Dense(2 => 10, relu),
        Dense(10 => 20, relu),
        Dense(20 => 5, relu),
        Dense(5 => 1)
    )
end

solver_params = SolverParams(true, 0, false, 0)

@time jump_model, upper_bounds, lower_bounds = NN_to_MIP(model, [1.0, 1.0], [-1.0, -1.0], solver_params; tighten_bounds=true);
@time jump_model_relax, upper_bounds_relax, lower_bounds_relax = NN_to_MIP(model, [1.0, 1.0], [-1.0, -1.0], solver_params; tighten_bounds=true);
vec(model(x_train)) ≈ [forward_pass!(jump_model, x_train[:, i])[1] for i in 1:750]
vec(model(x_train)) ≈ [forward_pass!(jump_model_relax, x_train[:, i])[1] for i in 1:750]

rmprocs(workers())

include("bound_tightening.jl")

n_neurons = 2 + sum(map(x -> length(x), [Flux.params(model)[2*k] for k in 1:length(model)]));
@time U, L = bound_tightening(model, [i<=2 ? 1.0 : 1000.0 for i in 1:n_neurons], [i<=2 ? -1.0 : -1000.0 for i in 1:n_neurons])
@time U_rel, L_rel = bound_tightening(model, [i<=2 ? 1.0 : 1000.0 for i in 1:n_neurons], [i<=2 ? -1.0 : -1000.0 for i in 1:n_neurons], false, true)

using Plots
plot(collect(Iterators.flatten(upper_bounds)) .- U[3:end-1])
plot!(collect(Iterators.flatten(lower_bounds)) .+ L[3:end-1])

plot(collect(Iterators.flatten(upper_bounds_relax)) .- U_rel[3:end-1])
plot!(collect(Iterators.flatten(lower_bounds_relax)) .+ L_rel[3:end-1])

plot(collect(Iterators.flatten(upper_bounds_relax)) - collect(Iterators.flatten(upper_bounds)))
plot!(collect(Iterators.flatten(lower_bounds_relax)) - collect(Iterators.flatten(lower_bounds)))