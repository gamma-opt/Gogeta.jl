data = rand(Float32, (2, 1000)) .- 0.5f0;
x_train = data[:, 1:750];

using Flux
using Random
using Revise
using Gogeta

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

begin
    Random.seed!(1234);

    model = Chain(
        Dense(2 => 10, relu),
        Dense(10 => 20, relu),
        Dense(20 => 5, relu),
        Dense(5 => 1)
    )
end

begin
    Random.seed!(1234);

    model = Chain(
        Dense(2 => 3, relu),
        Dense(3 => 1)
    )
end

solver_params = SolverParams(silent=true, threads=0, relax=false, time_limit=0)

@time jump, removed, compressed, U_comp, L_comp = compress(model, [1.0, 1.0], [-1.0, -1.0], solver_params);

vec(model(x_train)) ≈ [forward_pass!(jump, x_train[:, i])[1] for i in 1:750]
compressed(x_train) ≈ model(x_train)

@time _, U_full, L_full = NN_to_MIP(model, [1.0, 1.0], [-1.0, -1.0], solver_params; tighten_bounds=true);

using Plots

x = LinRange(-1.5, -0.5, 100)
y = LinRange(-0.5, 0.5, 100)

contourf(x, y, (x, y) -> model(hcat(x, y)')[], c=cgrad(:viridis), lw=0)
contourf(x, y, (x, y) -> compressed(hcat(x, y)')[], c=cgrad(:viridis), lw=0)

contourf(x, y, (x, y) -> forward_pass!(jump_model, vec(hcat(x, y)) .|> Float32)[], c=cgrad(:viridis), lw=0)
contourf(x, y, (x, y) -> forward_pass!(jump, vec(hcat(x, y)) .|> Float32)[], c=cgrad(:viridis), lw=0)

plot(collect(Iterators.flatten(U_comp[1:end-1])))
plot!(collect(Iterators.flatten(U_full[1:end-1])))

subdir_path = joinpath(parent_dir, subdirs[1])
model = load_model(n_neurons, subdir_path)

solver_params = SolverParams(silent=true, threads=0, relax=false, time_limit=0)
@time jump, removed, compressed, U_comp, L_comp = compress(model, [-0.5, 0.5], [-1.5, -0.5], solver_params; big_M=1_000_000.0);

U_data, L_data = get_bounds(subdir_path)
U_data = U_data[3:end]
L_data = L_data[3:end]

"""
"""

b = [Flux.params(model)[2*k] for k in 1:length(model)]
neuron_count = [length(b[k]) for k in eachindex(b)]

U_full = Vector{Vector}(undef, length(model))
L_full = Vector{Vector}(undef, length(model))

[U_full[layer] = Vector{Float64}(undef, neuron_count[layer]) for layer in 1:length(model)]
[L_full[layer] = Vector{Float64}(undef, neuron_count[layer]) for layer in 1:length(model)]

for layer in 1:length(model)
    for neuron in 1:neuron_count[layer]
        U_full[layer][neuron] = U_data[neuron + (layer == 1 ? 0 : cumsum(neuron_count)[layer-1])]
        L_full[layer][neuron] = L_data[neuron + (layer == 1 ? 0 : cumsum(neuron_count)[layer-1])]
    end
end

collect(Iterators.flatten(U_full)) == U_data
collect(Iterators.flatten(L_full)) == L_data

@time _, removed, compressed, U_comp, L_comp = compress(model, [-0.5, 0.5], [-1.5, -0.5], U_full, L_full);

solver_params = SolverParams(silent=true, threads=0, relax=false, time_limit=0)

@time compression_results = compress_fast(model, [-0.5, 0.5], [-1.5, -0.5], solver_params);
jump_model, removed_neurons, compressed_model, bounds_U, bounds_L = compression_results;

@time bound_results = NN_to_MIP(model, [-0.5, 0.5], [-1.5, -0.5], solver_params; tighten_bounds=true, big_M=100_000.0);

bound_results[3]

@time bound_compression = compress(model, [-0.5, 0.5], [-1.5, -0.5], bounds_U, bounds_L);

@time bound_results = NN_to_MIP(compressed_model, [-0.5, 0.5], [-1.5, -0.5], solver_params; tighten_bounds=true, big_M=100_000.0);

bound_results[1]

contourf(x, y, (x, y) -> forward_pass!(jump_model, vec(hcat(x, y)) .|> Float32)[], c=cgrad(:viridis), lw=0)
contourf(x, y, (x, y) -> model(hcat(x, y)')[], c=cgrad(:viridis), lw=0)
contourf(x, y, (x, y) -> compressed_model(hcat(x, y)')[], c=cgrad(:viridis), lw=0)

x1 = rand(Float32, 10) .- 1.5
x2 = rand(Float32, 10) .- 0.5
x = hcat(x1, x2)'

vec(model(x)) ≈ [forward_pass!(jump, x[:, i] .|> Float32) for i in 1:size(x)[2]]
vec(compressed(x)) ≈ [forward_pass!(jump, x[:, i] .|> Float32)[] for i in 1:size(x)[2]]
compressed(x) ≈ model(x)

@time jump_model, U_full, L_full = NN_to_MIP(model, [-0.5, 0.5], [-1.5, -0.5], solver_params; tighten_bounds=true, big_M=100_000.0);