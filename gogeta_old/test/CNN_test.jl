using Logging, Test
using ML_as_MO
using Random
using Flux, JuMP

# include("../src/nn/CNN_JuMP_model.jl") # REMOVE THIS WHEN ADDED TO PACKAGE

@info "Creating an arbitrary large convolutional network designed for 32x32 RGB images.
       We represent the CNN as a MILP using create_CNN_JuMP_model and assign the variables
       corresponging to the input layer to be same as our test image using evaluate_CNN!."

model = Chain(
    MaxPool((2,1)),
    MaxPool((1,2)),
    Conv((5,3), 3=>16, relu),
    Conv((3,5), 16=>32, relu),
    MeanPool((3,1)),
    MeanPool((1,3)),
    Conv((3,3), 32=>64, relu),
    MaxPool((2,2)),
    Flux.flatten,
    Dense(576, 64, relu),
    Dense(64, 32, relu),
    Dense(32, 16, relu),
    Dense(16, 10),
)

# random 64x64 RGB image
data = rand32(64, 64, 3, 1)

CNN_output = model(data)

# big-M values used for constraint bounds in the MILP
L_bounds = Vector{Array{Float32}}(undef, length(model))
U_bounds = Vector{Array{Float32}}(undef, length(model))

# bounds are set to [0,1] to the input layer (pixel values), and [-1000,1000] in the other layers (arbitrary large big-M)
L_bounds[1] = fill(0, (3,64,64));      U_bounds[1] = fill(1, (3,64,64))
L_bounds[2] = fill(-1000, (3,32,64));  U_bounds[2] = fill(1000, (3,32,64))
L_bounds[3] = fill(-1000, (3,32,32));  U_bounds[3] = fill(1000, (3,32,32))
L_bounds[4] = fill(-1000, (16,28,30)); U_bounds[4] = fill(1000, (16,28,30))
L_bounds[5] = fill(-1000, (32,26,26)); U_bounds[5] = fill(1000, (32,26,26))
L_bounds[6] = fill(-1000, (32,8,26));  U_bounds[6] = fill(1000, (32,8,26))
L_bounds[7] = fill(-1000, (32,8,8));   U_bounds[7] = fill(1000, (32,8,8))
L_bounds[8] = fill(-1000, (64,6,6));   U_bounds[8] = fill(1000, (64,6,6))
L_bounds[9] = fill(-1000, (64,3,3));   U_bounds[9] = fill(1000, (64,3,3))
L_bounds[10] = fill(-1000, (64,1,1));  U_bounds[10] = fill(1000, (64,1,1))
L_bounds[11] = fill(-1000, (32,1,1));  U_bounds[11] = fill(1000, (32,1,1))
L_bounds[12] = fill(-1000, (16,1,1));  U_bounds[12] = fill(1000, (16,1,1))
L_bounds[13] = fill(-1000, (10,1,1));  U_bounds[13] = fill(1000, (10,1,1))

# generating the MILP model and evaluating it with our input
MILP_model = create_CNN_JuMP_model(model, size(data), L_bounds, U_bounds)
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