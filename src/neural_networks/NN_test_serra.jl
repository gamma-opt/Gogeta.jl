init_ub = [1.0, 1.0]
init_lb = [-1.0, -1.0]

data = rand(Float32, (2, 1000)) .- 0.5f0;
x_train = data[:, 1:750];
y_train = [sum(x_train[:, col].^2) for col in 1:750];

using Distributed

include("bound_tightening_serra.jl")
include("bound_tightening_serra_parallel.jl")

addprocs(7)
@everywhere include("bound_tightening_serra_parallel.jl")
@everywhere gurobi_env = [Gurobi.Env() for i in 1:8];
gurobi_env = [Gurobi.Env() for i in 1:8];

using Random
Random.seed!(1234);

model = Chain(
    Dense(2 => 10, relu),
    Dense(10 => 30, relu),
    Dense(30 => 20, relu),
    Dense(20 => 5, relu),
    Dense(5 => 1)
);

@time jump_model, bounds_x, bounds_s = NN_to_MIP(model, init_ub, init_lb; tighten_bounds=true);
jump_model
bounds_x
bounds_s
@time [forward_pass!(jump_model, x_train[:, i])[1] for i in 1:750];
vec(model(x_train)) ≈ [forward_pass!(jump_model, x_train[:, i])[1] for i in 1:750]

include("bound_tightening.jl")

@time U, L = bound_tightening(model, [i<=2 ? 1.0 : 1000.0 for i in 1:2+10+30+20+5+1], [i<=2 ? -1.0 : -1000.0 for i in 1:2+10+30+20+5+1])

using Plots
plot(collect(Iterators.flatten(bounds_x)))
plot!(collect(Iterators.flatten(bounds_s)))
plot(collect(Iterators.flatten(bounds_x)) .- U[3:end-1])

@time jump_model, bounds_x, bounds_s = NN_to_MIP(model, init_ub, init_lb; tighten_bounds=false);
jump_model
bounds_x
bounds_s
@time [forward_pass!(jump_model, x_train[:, i])[1] for i in 1:750];
vec(model(x_train)) ≈ [forward_pass!(jump_model, x_train[:, i])[1] for i in 1:750]