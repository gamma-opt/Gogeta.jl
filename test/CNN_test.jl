using Logging, Test
using ML_as_MO
using Random
using Flux, JuMP

include("../src/nn/CNN_JuMP_model.jl") # REMOVE THIS WHEN ADDED TO PACKAGE

@info "Creating an arbitrary large convolutional network designed for 32x32 RGB images.
       We represent the CNN as a MILP using create_CNN_JuMP_model and assign the variables
       corresponging to the input layer to be same as our test image using evaluate_CNN!."

model = Chain(
    Conv((5,3), 3=>16, relu),
    Conv((3,5), 16=>32, relu),
    MeanPool((3,1)),
    MeanPool((1,3)),
    Conv((3,3), 32=>64, relu),
    MeanPool((2,2)),
    Flux.flatten,
    Dense(576, 64, relu),
    Dense(64, 32, relu),
    Dense(32, 16, relu),
    Dense(16, 10),
)

# random 32x32 RGB image
data = rand32(32, 32, 3, 1)

CNN_output = model(data)

# generating the MILP model and evaluating it with our input
MILP_model = create_CNN_JuMP_model(model, (32, 32, 3, 1), "image")
set_optimizer_attribute(MILP_model, "TimeLimit", 600)
evaluate_CNN!(MILP_model, data)
optimize!(MILP_model)

# extracting the values from the variables corresponging to the CNN output
x = MILP_model[:x]
MILP_output = zeros(Float32, 10)
len = length(model)
for i in 1:10
    MILP_output[i] = value(x[len-1,i,1,1])
end

@info "Tesing that both the CNN and the MILP model output (approximately) the same values"

# testing that the output values from the CNN and the MILP are (almost) the same
for i in 1:10
    @test MILP_output[i] ≈ CNN_output[i]
end

@test true