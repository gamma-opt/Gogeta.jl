using Logging
using Random
using Flux, JuMP

@info "Creating an arbitrary large deep network designed for 28x28 grayscale images.
       We represent the DNN as a MILP using create_JuMP_model and assign the variables
       corresponging to the input layer to be same as our test image using evaluate!."

model = Chain(
    Dense(784, 64, relu),
    Dense(64, 32, relu),
    Dense(32, 16, relu),
    Dense(16, 8, relu),
    Dense(8, 10),
)

# random flattened 28x28 grayscale image
data = rand32(784)

DNN_output = model(data)

# bounds are set to [0,1] to the input layer (pixel values), and [-1000,1000] in the other layers (arbitrary large big-M)
U_bounds = Float32[if i <= 784 1 else 1000 end for i in 1:914]
L_bounds = Float32[if i <= 784 0 else -1000 end for i in 1:914]

# generating the MILP model and evaluating it with our input
MILP_model = create_JuMP_model(model, L_bounds, U_bounds, "none", true)
evaluate!(MILP_model, data)
optimize!(MILP_model)

# extracting the values from the variables corresponging to the CNN output
x = MILP_model[:x]
MILP_output = zeros(Float32, 10)
len = length(model)
for i in 1:10
    MILP_output[i] = value(x[len,i])
end

@info "Tesing that both the CNN and the MILP model output (approximately) the same values"

# testing that the output values from the DNN and the MILP are (almost) the same
for i in 1:10
    @test MILP_output[i] ≈ DNN_output[i]
end

@test true