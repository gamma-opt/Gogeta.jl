using Flux
using GLPK
using JuMP
using LinearAlgebra
using Test 
using Statistics

@info "Creating a random neural network model"

dimension = 4

Random.seed!(1234)
NN_model = Chain(
    Dense(dimension, 16, relu),
    Dense(16, 8, relu),
    Dense(8, 1),
)

# Set upper and lower input bounds
init_U = ones(dimension);
init_L = zeros(dimension);

@info "Test that activation function at each layer (except last one) is relu"
K = length(NN_model)
@test all([NN_model[i].σ == relu for i in 1:K-1]) == true

@info "Test that activation function at the last layer is identity"
@test (NN_model[K].σ == identity) == true 

@info "Test that non-providing boundaries with 'precomputed' bound-tightening results in error"
@test_throws AssertionError NN_formulate_Psplit!(jump_model, NN_model, 4, init_U, init_L, bound_tightening = "precomputed")

@info "Test that non-existent strategy result in error"
@test_throws AssertionError NN_formulate_Psplit!(jump_model, NN_model, 4, init_U, init_L, strategy="hello_world")

@info "Test that P<2 with 'equalrange' strategy result in error"
@test_throws AssertionError NN_formulate_Psplit!(jump_model, NN_model, 2, init_U, init_L, strategy="equalrange")

dimension = 2

init_U = ones(dimension);
init_L = zeros(dimension);

@info "Test: dimension of boundaries should be the same as input layer in NN"
@test_throws AssertionError NN_formulate_Psplit!(jump_model, NN_model, 2, init_U, init_L)

dimension = 4
init_U = ones(dimension);
init_L = zeros(dimension);

@info "Generating sample dataset"
num_samples = 100
x = transpose(rand(100, dimension).* transpose(init_U .- init_L).+ transpose(init_L)) .|> Float32;

@info "Formulating the MIP with with strategy='equalsize' and bound_tightening='fast'"
jump_model = Model(GLPK.Optimizer);
set_silent(jump_model)
NN_formulate_Psplit!(jump_model, NN_model, 3, init_U, init_L; strategy="equalsize", bound_tightening="fast");

@info "Testing that corresponding JuMP model has the same output as NN, P=3"
@test vec(NN_model(x)) ≈ [forward_pass!(jump_model, input)[] for input in eachcol(x)]

@info "Testing that corresponding JuMP model has the same output as NN, P=4"
NN_formulate_Psplit!(jump_model, NN_model, 4, init_U, init_L; strategy="equalsize", bound_tightening="fast");
@test vec(NN_model(x)) ≈ [forward_pass!(jump_model, input)[] for input in eachcol(x)]

NN_formulate_Psplit!(jump_model, NN_model, 5, init_U, init_L; strategy="equalsize", bound_tightening="fast");
@info "Testing that corresponding JuMP model has the same output as NN, P=5"
@test vec(NN_model(x)) ≈ [forward_pass!(jump_model, input)[] for input in eachcol(x)]

@info "Formulating the MIP with with strategy='equalsize' and bound_tightening='standard', P=4"
NN_formulate_Psplit!(jump_model, NN_model, 4, init_U, init_L, bound_tightening = "standard", strategy = "equalsize");
@info "Testing that corresponding JuMP model has the same output as NN,  strategy = 'equalsize'"
@test vec(NN_model(x)) ≈ [forward_pass!(jump_model, input)[] for input in eachcol(x)]

@info "Testing that corresponding JuMP model has the same output as NN,  strategy = 'equalrange'"
NN_formulate_Psplit!(jump_model, NN_model, 4, init_U, init_L, bound_tightening = "standard", strategy = "equalrange");
@test vec(NN_model(x)) ≈ [forward_pass!(jump_model, input)[] for input in eachcol(x)]

@info "Testing that corresponding JuMP model has the same output as NN,  strategy = 'snake'"
NN_formulate_Psplit!(jump_model, NN_model, 4, init_U, init_L, bound_tightening = "standard", strategy = "snake");
@test vec(NN_model(x)) ≈ [forward_pass!(jump_model, input)[] for input in eachcol(x)]

@info "Testing that corresponding JuMP model has the same output as NN,  strategy = 'random'"
NN_formulate_Psplit!(jump_model, NN_model, 4, init_U, init_L, bound_tightening = "standard", strategy = "random");
@test vec(NN_model(x)) ≈ [forward_pass!(jump_model, input)[] for input in eachcol(x)]