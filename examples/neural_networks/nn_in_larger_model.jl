using Flux
using Random
using Gogeta
using Plots
using Gurobi
using JuMP

# Create a small neural network with random weights
begin
    Random.seed!(1234);

    NN_model = Chain(
        Dense(2 => 10, relu),
        Dense(10 => 50, relu),
        Dense(50 => 20, relu),
        Dense(20 => 5, relu),
        Dense(5 => 1)
    )
end

# Set upper and lower input bounds
init_U = [-0.5, 0.5];
init_L = [-1.5, -0.5];

# contour plot the model in the feasible region of the following optimization problem
x_range = LinRange{Float32}(init_L[1], init_U[1], 100);
y_range = LinRange{Float32}(init_L[2], init_U[2], 100);

contourf(x_range, y_range, (x, y) -> if y >= -x-1 NN_model(hcat(x, y)')[] else NaN end, c=cgrad(:viridis), lw=0)

# create an optimization problem
jump_model = Model(Gurobi.Optimizer)

@variable(jump_model, -1.5 <= x <= -0.5)
@variable(jump_model, -0.5 <= y <= 0.5)
@variable(jump_model, output)

@constraint(jump_model, y >= -x - 1)

@objective(jump_model, Max, output - 0.5*x)

# set up a solver for bound tightening
ENV = Gurobi.Env()
function set_solver!(jump)
    set_optimizer(jump, () -> Gurobi.Optimizer(ENV))
    #relax_integrality(jump) # Use this to solve bounds with binary variables relaxed. Looser bounds but faster bound tightening.
    set_silent(jump)
end

# include neural network as a part of the larger optimization problem
# use bound tightening and compress the network
NN_incorporate!(jump_model, NN_model, output, x, y; U_in=init_U, L_in=init_L, compress=true, bound_tightening="standard")

optimize!(jump_model)
solution_summary(jump_model)

# see optimal solution
value(x)
value(y)
value(output)

# check that NN formulation matches Flux model
set_silent(jump_model)
check = [if y_in >= -x_in-1 forward_pass_NN!(jump_model, [x_in, y_in], output, x, y) ≈ NN_model(hcat(x_in, y_in)')[] end for x_in in x_range, y_in in y_range]
all(value -> value === nothing || value == true, check)