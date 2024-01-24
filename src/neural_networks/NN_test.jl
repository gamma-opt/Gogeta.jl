using Distributed

addprocs(2, exeflags=["--project"])

@everywhere include("bound_tightening_new.jl")
#BSON.@load string(@__DIR__)*"/NN_paraboloid.bson" model

init_ub = [1.0, 1.0]
init_lb = [-1.0, -1.0]

data = rand(Float32, (2, 1000)) .- 0.5f0;
x_train = data[:, 1:750];
y_train = [sum(x_train[:, col].^2) for col in 1:750];

model = Chain(
    Dense(2 => 30, relu),
    Dense(30 => 50, relu),
    Dense(50 => 1, relu)
)

const ENV = Gurobi.Env();

@time jump_model, bounds_x, bounds_s = NN_to_MIP(model, init_ub, init_lb, ENV; tighten_bounds=true);
jump_model
bounds_x
bounds_s
@time [forward_pass!(jump_model, x_train[:, i])[1] for i in 1:750];
vec(model(x_train)) ≈ [forward_pass!(jump_model, x_train[:, i])[1] for i in 1:750]

@time jump_model, bounds_x, bounds_s = NN_to_MIP(model, init_ub, init_lb, ENV; tighten_bounds=false);
jump_model
bounds_x
bounds_s
@time [forward_pass!(jump_model, x_train[:, i])[1] for i in 1:750];
vec(model(x_train)) ≈ [forward_pass!(jump_model, x_train[:, i])[1] for i in 1:750]
