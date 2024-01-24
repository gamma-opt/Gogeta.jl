include("bound_tightening_new.jl")
BSON.@load string(@__DIR__)*"/NN_paraboloid.bson" model

init_ub = [1.0, 1.0]
init_lb = [-1.0, -1.0]

model = Chain(
    Dense(2 => 30, relu),
    Dense(30 => 50, relu),
    Dense(50 => 1, relu)
) 

x_correct = [
    Float32[1.3650875, 1.6169983, 1.1242903, 1.3936517],
    Float32[0.9487423, -0.0, 0.8809887],
    Float32[1.0439509]
]

s_correct = [
    Float32[1.6752996, 2.0683866, 0.40223768, 1.4051342],
    Float32[0.36901128, 0.76332575, 0.47233626],
    Float32[-0.0]
]

const ENV = Gurobi.Env();

@time jump_model, bounds_x, bounds_s = NN_to_MIP(model, init_ub, init_lb, ENV; tighten_bounds=true);
jump_model
bounds_x
bounds_s

data = rand(Float32, (2, 1000)) .- 0.5f0;
x_train = data[:, 1:750];
y_train = [sum(x_train[:, col].^2) for col in 1:750];
@time [forward_pass!(jump_model, x_train[:, i])[1] for i in 1:750];
vec(model(x_train)) ≈ [forward_pass!(jump_model, x_train[:, i])[1] for i in 1:750]