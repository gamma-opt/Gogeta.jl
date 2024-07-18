using Random
using Gogeta
using Plots
using Gurobi
using JuMP

# create an optimization problem
jump_model = Model(Gurobi.Optimizer)

@variable(jump_model, -1 <= x <= 1)
@variable(jump_model, -1 <= y <= 1)
@variable(jump_model, output)

@constraint(jump_model, y >= 1-x)

@objective(jump_model, Min, x+y)

# include input convex neural network as a part of the larger optimization problem
ICNN_incorporate!(jump_model, "model_weights.json", output, x, y)

optimize!(jump_model)
solution_summary(jump_model)

# see optimal solution
value(x)
value(y)
value(output)

# plot the icnn by itself
jump_model = Model(Gurobi.Optimizer)

@variable(jump_model, -1 <= x <= 1)
@variable(jump_model, -1 <= y <= 1)
@variable(jump_model, output)
@objective(jump_model, Min, 0)

ICNN_incorporate!(jump_model, "model_weights.json", output, x, y)

x_range = LinRange{Float32}(-1, 1, 100);
y_range = LinRange{Float32}(-1, 1, 100);

contourf(x_range, y_range, (x_in, y_in) -> forward_pass_ICNN!(jump_model, [x_in, y_in], output, x, y), c=cgrad(:viridis), lw=0)