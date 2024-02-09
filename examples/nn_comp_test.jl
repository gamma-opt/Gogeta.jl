using Flux
using Random
using Revise
using Gogeta
using Test
using Plots

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

init_U = [-0.5, 0.5];
init_L = [-1.0, -1.0];

x1 = (rand(100) * (init_U[1] - init_L[1])) .+ init_L[1];
x2 = (rand(100) * (init_U[2] - init_L[2])) .+ init_L[2];
x = transpose(hcat(x1, x2)) .|> Float32;

solver_params = SolverParams(silent=true, threads=0, relax=false, time_limit=0);

# compress with compress fast
@time compression_results = compress_fast(model, init_U, init_L, solver_params);
jump_model, removed_neurons, compressed_model, bounds_U, bounds_L = compression_results;

# test that jump model and compressed model produce same results as original
@test vec(model(x)) ≈ [forward_pass!(jump_model, x[:, i])[] for i in 1:size(x)[2]]
@test compressed_model(x) ≈ model(x)

# perform bound tightening
@time nn_jump, U_correct, L_correct = NN_to_MIP(model, init_U, init_L, solver_params; tighten_bounds=true);

# test that created jump model is equal to the original
@test vec(model(x)) ≈ [forward_pass!(nn_jump, x[:, i])[] for i in 1:size(x)[2]]

# compare bounds with/without fast compression
plot(collect(Iterators.flatten(bounds_U[1:end-1])))
plot!(collect(Iterators.flatten(U_correct[1:end-1])))

plot!(collect(Iterators.flatten(bounds_L[1:end-1])))
plot!(collect(Iterators.flatten(L_correct[1:end-1])))

# contour plot the model
x_range = LinRange(init_L[1], init_U[1], 100);
y_range = LinRange(init_L[2], init_U[2], 100);

contourf(x_range, y_range, (x, y) -> model(hcat(x, y)')[], c=cgrad(:viridis), lw=0)
contourf(x_range, y_range, (x, y) -> compressed_model(hcat(x, y)')[], c=cgrad(:viridis), lw=0)