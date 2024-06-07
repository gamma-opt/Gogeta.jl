using JuMP
using GLPK
using Plots
using Gogeta

icnn_jump = Model(GLPK.Optimizer)

input_vars, output_var = ICNN_formulate!(icnn_jump, "examples/model_weights.json");

##### PLOTTING #####

init_U = [1.0, 1.0];
init_L = [-1.0, -1.0];

x_range = LinRange{Float32}(init_L[1], init_U[1], 50);
y_range = LinRange{Float32}(init_L[2], init_U[2], 50);

f(x, y) = x^2 + y^2

plot(x_range, y_range, f, st=:surface, c=cgrad(:viridis), camera=(-40, 50))

plot(x_range, y_range, (x, y) -> forward_pass_ICNN!(icnn_jump, [x, y]), st=:surface, c=cgrad(:viridis), camera=(-40, 50))