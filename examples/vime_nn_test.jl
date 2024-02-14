using Flux
using Random
using Revise
using Gogeta
using Test
using Plots
using NPZ

n_neurons = Int64[2, 1024, 512, 512, 256, 1]
n_neurons_cumulative_indices = [i+1 for i in [0, cumsum(n_neurons)...]]
model = load_model(n_neurons, "/Users/eetureijonen/Desktop/GAMMA-OPT/Gogeta.jl/src/neural_networks/compression/layer_weights/model_Adadelta_0.001_0.001_0")

init_U = [-0.5, 0.5];
init_L = [-1.5, -0.5];

x1 = (rand(10) * (init_U[1] - init_L[1])) .+ init_L[1];
x2 = (rand(10) * (init_U[2] - init_L[2])) .+ init_L[2];
x = transpose(hcat(x1, x2)) .|> Float32;

solver_params = SolverParams(solver="Gurobi", silent=true, threads=0, relax=false, time_limit=0);

# test jump model using only fast bound tightening
@time nn_loose, U_loose, L_loose = NN_to_MIP(model, init_U, init_L, solver_params; tighten_bounds=false);

# test that created jump model is equal to the original (0.1% tolerance)
@test isapprox(vec(model(x)), [forward_pass!(nn_loose, x[:, i])[] for i in 1:size(x)[2]]; rtol=0.001)

plot(relu.(collect(Iterators.flatten(U_loose))[1:end-1]) .+ 0.01, 
    yscale=:log10, 
    yticks=([1, 100, 500, 1000, 2000], string.([1, 100, 500, 1000, 2000])), 
    legend=false
)

U_data, L_data = get_bounds("/Users/eetureijonen/Desktop/GAMMA-OPT/Gogeta.jl/src/neural_networks/compression/layer_weights/model_Adadelta_0.001_0.001_0");
U_data = U_data[3:end];
L_data = L_data[3:end];

plot!(relu.(U_data[1:end-1]) .+ 0.01)


# compress with precomputed loose bounds
@time compressed_loose, removed_loose = compress(model, init_U, init_L; bounds_U=U_loose, bounds_L=L_loose);

# test that compressed model is same as the original
@test compressed_loose(x) ≈ model(x)

b = [Flux.params(model)[2*k] for k in 1:length(model)];
neuron_count = [length(b[k]) for k in eachindex(b)];

U_full = Vector{Vector}(undef, length(model));
L_full = Vector{Vector}(undef, length(model));

[U_full[layer] = Vector{Float64}(undef, neuron_count[layer]) for layer in 1:length(model)];
[L_full[layer] = Vector{Float64}(undef, neuron_count[layer]) for layer in 1:length(model)];

for layer in 1:length(model)
    for neuron in 1:neuron_count[layer]
        U_full[layer][neuron] = U_data[neuron + (layer == 1 ? 0 : cumsum(neuron_count)[layer-1])]
        L_full[layer][neuron] = L_data[neuron + (layer == 1 ? 0 : cumsum(neuron_count)[layer-1])]
    end
end

collect(Iterators.flatten(U_full)) == U_data
collect(Iterators.flatten(L_full)) == L_data

# compress with precomputed LP bounds
@time res = compress(model, init_U, init_L; bounds_U=U_full, bounds_L=L_full);
compressed, removed = res;

# test that compressed model is same as the original
@test compressed(x) ≈ model(x)

# contour plot the model
x_range = LinRange{Float32}(init_L[1], init_U[1], 100);
y_range = LinRange{Float32}(init_L[2], init_U[2], 100);

contourf(x_range, y_range, (x, y) -> model(hcat(x, y)')[], c=cgrad(:viridis), lw=0)
contourf(x_range, y_range, (x, y) -> compressed(hcat(x, y)')[], c=cgrad(:viridis), lw=0)
