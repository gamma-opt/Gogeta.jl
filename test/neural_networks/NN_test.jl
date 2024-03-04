using Flux
using Random
using Gogeta
using GLPK
using JuMP

@info "Creating a medium-sized neural network with random weights"
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

init_U = [-0.5, 0.5];
init_L = [-1.5, -0.5];

x1 = (rand(100) * (init_U[1] - init_L[1])) .+ init_L[1];
x2 = (rand(100) * (init_U[2] - init_L[2])) .+ init_L[2];
x = transpose(hcat(x1, x2)) .|> Float32;

@info "Compressing the neural network with simultaneous bound tightening."
jump_model = Model()
set_optimizer(jump_model, GLPK.Optimizer)
set_silent(jump_model)
set_attribute(jump_model, "tm_lim", 1.0 * 1_000)
compressed_model, removed_neurons, bounds_U, bounds_L = NN_formulate!(jump_model, NN_model, init_U, init_L; bound_tightening="standard", compress=true, silent=false);

@info "Testing that the compressed model and the corresponding JuMP model are equal to the original neural network."
@test vec(NN_model(x)) ≈ [forward_pass!(jump_model, input)[] for input in eachcol(x)]
@test compressed_model(x) ≈ NN_model(x)

@info "Creating a JuMP model from the neural network with bound tightening but without compression."
nn_jump = Model()
set_optimizer(nn_jump, GLPK.Optimizer)
set_silent(nn_jump)
set_attribute(nn_jump, "tm_lim", 1.0 * 1_000)
U_correct, L_correct = NN_formulate!(nn_jump, NN_model, init_U, init_L; bound_tightening="standard", silent=false)

@info "Creating bound tightened JuMP model with output bounds present."
jump_nor = Model()
set_optimizer(jump_nor, GLPK.Optimizer)
set_silent(jump_nor)
set_attribute(jump_nor, "tm_lim", 1.0 * 1_000)
U_nor, L_nor = NN_formulate!(jump_nor, NN_model, init_U, init_L; bound_tightening="output", U_out=[-0.2], L_out=[-0.4], silent=false);

@info "Testing that the output tightened model is the same in the areas it's defined in."
@test all(map(input -> begin
    test = forward_pass!(jump_nor, input)
    if !isnan(test[])
        test[] ≈ NN_model(input)[]
    else
        true
    end
end, eachcol(x)))

@info "Testing that the created JuMP model is equal to the original neural network."
@test vec(NN_model(x)) ≈ [forward_pass!(nn_jump, input)[] for input in eachcol(x)]

@info "Compressing with the precomputed bounds."
compressed, removed = NN_compress(NN_model, init_U, init_L, U_correct, L_correct);

@info "Testing that this compression result is equal to the original."
@test compressed(x) ≈ NN_model(x)

@info "Testing that the removed neurons are the same as with simultaneous bound tightening compression."
@test removed == removed_neurons

@info "Creating a JuMP model of the network with loose bound tightening."
nn_loose = Model()
set_optimizer(nn_loose, GLPK.Optimizer)
set_silent(nn_loose)
set_attribute(nn_loose, "tm_lim", 1.0 * 1_000)
U_loose, L_loose = NN_formulate!(nn_loose, NN_model, init_U, init_L; silent=false)

@info "Testing that the loose JuMP model is equal to the original neural network."
@test vec(NN_model(x)) ≈ [forward_pass!(nn_loose, input)[] for input in eachcol(x)]

@info "Compressing with the precomputed loose bounds."
compressed_loose, removed_loose = NN_compress(NN_model, init_U, init_L, U_loose, L_loose);

@info "Testing that this loose compression result is equal to the original neural network."
@test compressed_loose(x) ≈ NN_model(x)
