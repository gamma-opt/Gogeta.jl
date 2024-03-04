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

# Formulate the MIP with optimization-based bound tightening
jump_model = Model(Gurobi.Optimizer);
set_silent(jump_model)
set_attribute(jump_model, "TimeLimit", 10)

@time bounds_U, bounds_L = NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="standard", silent=false);

# contour plot the model
x_range = LinRange{Float32}(init_L[1], init_U[1], 100);
y_range = LinRange{Float32}(init_L[2], init_U[2], 100);

contourf(x_range, y_range, (x, y) -> NN_model(hcat(x, y)')[], c=cgrad(:viridis), lw=0)
contourf(x_range, y_range, (x, y) -> forward_pass!(jump_model, [x, y])[], c=cgrad(:viridis), lw=0)

# Formulate the MIP with optimization-based bound tightening and compress
jump_model = Model(Gurobi.Optimizer);
set_silent(jump_model)
set_attribute(jump_model, "TimeLimit", 10)
@time NN_comp, removed, comp_U, comp_L = NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="standard", compress=true, silent=false);

# Plot compressed model - should be the same
contourf(x_range, y_range, (x, y) -> NN_comp(hcat(x, y)')[], c=cgrad(:viridis), lw=0)

# Formulate with fast/heuristic bound tightening
jump_model = Model(Gurobi.Optimizer);
set_silent(jump_model)
set_attribute(jump_model, "TimeLimit", 10)

@time loose_U, loose_L = NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="fast", silent=false);

# Compare fast tightened bounds with standard ones
plot(collect(Iterators.flatten(bounds_U)))
plot!(collect(Iterators.flatten(bounds_L)))

plot!(collect(Iterators.flatten(loose_U)))
plot!(collect(Iterators.flatten(loose_L)))

# Formulate with output bounds considered
jump_model = Model(Gurobi.Optimizer);
set_silent(jump_model)
@time out_U, out_L = NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="output", U_out=[-0.2], L_out=[-0.4], silent=false);

# Plot output bound model
contourf(x_range, y_range, (x, y) -> forward_pass!(jump_model, [x, y])[], c=cgrad(:viridis), lw=0)

# Compress with precomputed bounds
NN_precomp, removed_precomp = NN_compress(NN_model, init_U, init_L, bounds_U, bounds_L);

# Formulate with precomputed bounds
jump_model = Model(Gurobi.Optimizer);
set_silent(jump_model)
@time NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="precomputed", U_bounds=bounds_U, L_bounds=bounds_L);