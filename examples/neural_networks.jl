using Flux
using Random
using Gogeta
using Plots

# Create a small neural network with random weights
begin
    Random.seed!(1234);

    model = Chain(
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

# Set parameters for the solver
solver_params = SolverParams(solver="Gurobi", silent=true, threads=0, relax=false, time_limit=0);

# Compress the neural network with simultaneous bound tightening.
@time jump_model, compressed_model, removed_neurons, bounds_U, bounds_L = compress_with_bound_tightening(model, init_U, init_L,solver_params; bound_tightening="standard");

# contour plot the model
x_range = LinRange{Float32}(init_L[1], init_U[1], 100);
y_range = LinRange{Float32}(init_L[2], init_U[2], 100);

contourf(x_range, y_range, (x, y) -> model(hcat(x, y)')[], c=cgrad(:viridis), lw=0)
contourf(x_range, y_range, (x, y) -> compressed_model(hcat(x, y)')[], c=cgrad(:viridis), lw=0)
contourf(x_range, y_range, (x, y) -> forward_pass!(jump_model, [x, y])[], c=cgrad(:viridis), lw=0)

# Create a JuMP model from the neural network with bound tightening but without compression
@time nn_jump, U_correct, L_correct = NN_to_MIP_with_bound_tightening(model, init_U, init_L, solver_params; bound_tightening="standard");

# Creating bound tightened JuMP model with output bounds present.
@time jump_nor, U_nor, L_nor = NN_to_MIP_with_bound_tightening(model, init_U, init_L, solver_params; bound_tightening="output", U_out=[-0.2], L_out=[-0.4]);

# Compare tightened bounds with/without output bounds consideration
plot(collect(Iterators.flatten(U_correct)))
plot!(collect(Iterators.flatten(U_nor)))

plot!(collect(Iterators.flatten(L_correct)))
plot!(collect(Iterators.flatten(L_nor)))

# Compress with the precomputed bounds.
@time compressed, removed = compress_with_precomputed(model, init_U, init_L, U_correct, L_correct);

# Create a JuMP model of the network with fast bound tightening.
nn_loose, U_loose, L_loose = NN_to_MIP_with_bound_tightening(model, init_U, init_L, solver_params; bound_tightening="fast");

# Compare fast tightened bounds with standard ones
plot(collect(Iterators.flatten(U_correct)))
plot!(collect(Iterators.flatten(U_loose)))

plot!(collect(Iterators.flatten(L_correct)))
plot!(collect(Iterators.flatten(L_loose)))