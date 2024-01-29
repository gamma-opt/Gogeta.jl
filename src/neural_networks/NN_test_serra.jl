data = rand(Float32, (2, 1000)) .- 0.5f0;
x_train = data[:, 1:750];
y_train = [sum(x_train[:, col].^2) for col in 1:750];

using Distributed

include("bound_tightening_serra.jl")

addprocs(7)
@everywhere include("bound_tightening_serra_parallel.jl")

@everywhere GUROBI_ENV = [Gurobi.Env() for i in 1:nprocs()];
@everywhere SILENT = true;
@everywhere LIMIT = 0;
@everywhere RELAX = false;
@everywhere THREADS = 0;

using Random
Random.seed!(1234);

model = Chain(
    Dense(2 => 10, relu),
    Dense(10 => 30, relu),
    #Dense(30 => 30, relu),
    Dense(30 => 20, relu),
    Dense(20 => 5, relu),
    Dense(5 => 1)
);

@time jump_model, upper_bounds, lower_bounds = NN_to_MIP(model, [1.0, 1.0], [-1.0, -1.0]; tighten_bounds=true);
vec(model(x_train)) ≈ [forward_pass!(jump_model, x_train[:, i])[1] for i in 1:750]

include("bound_tightening.jl")

n_neurons = 2 + sum(map(x -> length(x), [Flux.params(model)[2*k] for k in 1:length(model)]));
@time U, L = bound_tightening(model, [i<=2 ? 1.0 : 1000.0 for i in 1:n_neurons], [i<=2 ? -1.0 : -1000.0 for i in 1:n_neurons])

using Plots
plot(collect(Iterators.flatten(upper_bounds)) .- U[3:end-1])
plot!(collect(Iterators.flatten(lower_bounds)) .+ L[3:end-1])