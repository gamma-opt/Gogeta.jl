using Flux
using Random
using Gogeta
using GLPK
using JuMP

@info "Creating a small neural network with random weights"
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

x_range = LinRange{Float32}(init_L[1], init_U[1], 100);
y_range = LinRange{Float32}(init_L[2], init_U[2], 100);

@info "Creating an optimization problem"
jump_model = Model(GLPK.Optimizer)

@variable(jump_model, -1.5 <= x <= -0.5)
@variable(jump_model, -0.5 <= y <= 0.5)
@variable(jump_model, output)

@constraint(jump_model, y >= -x - 1)

@objective(jump_model, Max, output - 0.5*x)

@info "setting up a solver for bound tightening"
function set_solver!(jump)
    set_optimizer(jump, GLPK.Optimizer)
    set_silent(jump)
end

@info "include neural network as a part of the larger optimization problem. Use bound tightening and compress the network"
NN_incorporate!(jump_model, NN_model, output, x, y; U_in=init_U, L_in=init_L, compress=true, bound_tightening="standard")

optimize!(jump_model)

@info "Testing that correct optimum is found"
@test value(x) ≈ -1.2609439405453133
@test value(y) ≈ 0.2609439405453132
@test value(output) ≈ -0.3214315186715632

@info "Testing that NN output matches Flux model"
set_silent(jump_model)
check = [if y_in >= -x_in-1 forward_pass_NN!(jump_model, [x_in, y_in], output, x, y) ≈ NN_model(hcat(x_in, y_in)')[] end for x_in in x_range, y_in in y_range]
@test all(value -> value === nothing || value == true, check)