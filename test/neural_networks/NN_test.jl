using Flux
using Random
using Gogeta

@info "Creating a medium-sized neural network with random weights"
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
init_L = [-1.5, -0.5];

x1 = (rand(100) * (init_U[1] - init_L[1])) .+ init_L[1];
x2 = (rand(100) * (init_U[2] - init_L[2])) .+ init_L[2];
x = transpose(hcat(x1, x2)) .|> Float32;

solver_params = SolverParams(solver="GLPK", silent=true, threads=0, relax=false, time_limit=0);

@info "Compressing the neural network with simultaneous bound tightening."
jump_model, compressed_model, removed_neurons, bounds_U, bounds_L = compress(model, init_U, init_L; params=solver_params);

@info "Testing that the compressed model and the corresponding JuMP model are equal to the original neural network."
@test vec(model(x)) ≈ [forward_pass!(jump_model, x[:, i])[] for i in 1:size(x)[2]]
@test compressed_model(x) ≈ model(x)

@info "Creating a JuMP model from the neural network with bound tightening but without compression."
nn_jump, U_correct, L_correct = NN_to_MIP(model, init_U, init_L, solver_params; tighten_bounds="standard");

@info "Testing that the created JuMP model is equal to the original neural network."
@test vec(model(x)) ≈ [forward_pass!(nn_jump, x[:, i])[] for i in 1:size(x)[2]]

@info "Compressing with the precomputed bounds."
compressed, removed = compress(model, init_U, init_L; bounds_U=U_correct, bounds_L=L_correct);

@info "Testing that this compression result is equal to the original."
@test compressed(x) ≈ model(x)

@info "Testing that the removed neurons are the same as with simultaneous bound tightening compression."
@test removed ≈ removed_neurons

@info "Creating a JuMP model of the network with loose bound tightening."
nn_loose, U_loose, L_loose = NN_to_MIP(model, init_U, init_L, solver_params; tighten_bounds="fast");

@info "Testing that the loose JuMP model is equal to the original neural network."
@test vec(model(x)) ≈ [forward_pass!(nn_loose, x[:, i])[] for i in 1:size(x)[2]]

@info "Compressing with the precomputed loose bounds."
compressed_loose, removed_loose = compress(model, init_U, init_L; bounds_U=U_loose, bounds_L=L_loose);

@info "Testing that this loose compression result is equal to the original neural network."
@test compressed_loose(x) ≈ model(x)