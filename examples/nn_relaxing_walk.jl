using Flux
using Random
using Gogeta
using Gurobi
using JuMP

dimension = 2

begin
    Random.seed!(1234);

    NN_model = Chain(
        Dense(dimension => 100, relu),
        Dense(100 => 100, relu),
        Dense(100 => 1)
    )
end

# Set upper and lower input bounds
init_U = ones(dimension);
init_L = zeros(dimension);

# Formulate the MIP with heuristic bound tightening
jump_model = Model(Gurobi.Optimizer)
set_silent(jump_model)
NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="fast");

last_layer, _ = maximum(keys(jump_model[:x].data))
@objective(jump_model, Max, jump_model[:x][last_layer, 1])

x_range = LinRange{Float32}(init_L[1], init_U[1], 100);
y_range = LinRange{Float32}(init_L[2], init_U[2], 100);

contourf(x_range, y_range, (x, y) -> NN_model(hcat(x, y)')[], c=cgrad(:viridis), lw=0)

for x in range(0, 1, 10), y in range(0, 1, 10)
    println("($x, $y)")
    _, path = local_search([x, y], jump_model, init_U, init_L)
    display(plot!(first.(path), last.(path), marker=:circle, legend=false))
end

_, path = local_search([0.1, 0.0], jump_model, init_U, init_L)

# optimize_by_walking!(jump_model, init_U, init_L)



function set_solver!(jump)
    set_optimizer(jump, Gurobi.Optimizer)
    set_silent(jump)
end