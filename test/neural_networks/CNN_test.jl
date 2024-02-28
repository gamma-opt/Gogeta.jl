using Flux
using Random
using Gogeta
using HiGHS
using JuMP

input = rand(Float32, 30, 20, 1, 1)

@info "Creating a random convolutional neural network model"

Random.seed!(1234)
CNN_model = Flux.Chain(
    Conv((4,3), 1 => 10, pad=(2, 1), stride=(3, 2), relu),
    MaxPool((6,4), pad=(1, 3), stride=(3, 2)),
    Flux.flatten,
    Dense(210 => 30, relu),
    Dense(30 => 1)
)

@info "Formulating as a JuMP model"

jump = Model(HiGHS.Optimizer)
set_silent(jump)
cnns = get_structure(CNN_model, input)
create_MIP_from_CNN!(jump, CNN_model, cnns)

@info "Testing image forward pass with some random inputs - passing test indicates that model is constructed correctly."

for _ in 1:10
    test_input = rand(Float32, 30, 20, 1, 1)
    @test vec(CNN_model(test_input)) ≈ image_pass!(jump, test_input)
end

@info "Repeating the test with a different model and input size"

input = rand(Float32, 15, 40, 1, 1)

@info "Creating a random convolutional neural network model"

Random.seed!(1234)
CNN_model = Flux.Chain(
    Conv((3,3), 1 => 5, pad=(1, 3), stride=(1, 2), relu),
    MaxPool((5,4), stride=(3, 2)),
    MeanPool((2,2), stride=(1, 2)),
    Flux.flatten,
    Dense(75 => 15, relu),
    Dense(15 => 1)
)

@info "Formulating as a JuMP model"

jump = Model(HiGHS.Optimizer)
set_silent(jump)
cnns = get_structure(CNN_model, input)
create_MIP_from_CNN!(jump, CNN_model, cnns)

@info "Testing image forward pass with some random inputs - passing test indicates that model is constructed correctly."

for _ in 1:10
    test_input = rand(Float32, 15, 40, 1, 1)
    @test vec(CNN_model(test_input)) ≈ image_pass!(jump, test_input)
end