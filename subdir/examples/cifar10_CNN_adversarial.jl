using Logging
using Statistics
using Flux
using Flux: onehotbatch, logitcrossentropy, train!
using MLDatasets
using MLDatasets: CIFAR10
using ML_as_MO

# include("../src/nn/CNN_JuMP_model.jl") # REMOVE THIS WHEN ADDED TO PACKAGE
# location of the create_CNN_adv function
include("helper_functions.jl")

# This file shows how to use the convert a CNN into a MILP using the create_CNN_JuMP_model function.

# The CIFAR10 dataset consists of 50 000 32x32x3 images of 10 different classes. The 3rd dimension is the RGB channel.
# The 10 classes are as follows: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

@info "Creating and training a small ReLU CNN using Flux.jl based on the CIFAR10 image dataset"

x_train, y_train = CIFAR10(split=:train)[:]
x_test,  y_test  = CIFAR10(split=:test)[:]

y_train_oh = onehotbatch(y_train, 0:9)
y_test_oh = onehotbatch(y_test, 0:9)

train = [(x_train, y_train_oh)]
test = [(x_test, y_test)]

model = Chain( # 4 epochs atm
    Conv((3,3), 3=>5, relu),
    MeanPool((2,2)),
    Conv((3,3), 5=>6, relu),
    MeanPool((2,2)),
    Flux.flatten,
    Dense(216, 24, relu),
    Dense(24, 10),
)

parameters = params(model)
loss(x, y) = logitcrossentropy(model(x), y)
opt = Adam(0.01)

@info "Training the CNN for 50 cycles with the full training set and printing its accuracy on the test set"

# training the CNN
n = 50
loss_values = zeros(n)
for i in 1:n
    train!(loss, parameters, train, opt)
    loss_values[i] = loss(x_train, y_train_oh)
    if i % 5 == 0
        println("Training cycle $i, loss value $(loss_values[i])")
    end
end

# calculating the accuracy of the CNN (around 35% with these training parameters)
correct_guesses = 0
test_len = length(y_test)
for i in 1:test_len
    if findmax(model(reshape(x_test[:,:,:,i], 32, 32, 3, 1)))[2][1] - 1 == y_test[i] # -1 to get right index
        correct_guesses += 1
    end
end
acc = correct_guesses / test_len
println("Accuracy: $acc")

@info "Creating a MILP model and generating one adversarial image based on the trained ReLU CNN.
       The adversarial image is generated by minimising the L1-norm difference between the original
       and the adversarial image such that the output node corresponging to the adversarial label
       has a 20% bigger value than in all other output nodes. Here, we require than an image of a
       \"truck\" is missclassified as a \"deer\". A timelimit of 600 sec is used. 
       (L2-norm can also be used but this requires larger computational time to give a solution)"

# big-M values used for constraint bounds in the MILP
L_bounds = Vector{Array{Float32}}(undef, length(model))
U_bounds = Vector{Array{Float32}}(undef, length(model))

L_bounds[1] = fill(0, (3,32,32));     U_bounds[1] = fill(1, (3,32,32))
L_bounds[2] = fill(-1000, (5,30,30)); U_bounds[2] = fill(1000, (5,30,30))
L_bounds[3] = fill(-1000, (5,15,15)); U_bounds[3] = fill(1000, (5,15,15))
L_bounds[4] = fill(-1000, (6,13,13)); U_bounds[4] = fill(1000, (6,13,13))
L_bounds[5] = fill(-1000, (6,6,6));   U_bounds[5] = fill(1000, (6,6,6))
L_bounds[6] = fill(-1000, (24,1,1));  U_bounds[6] = fill(1000, (24,1,1))
L_bounds[7] = fill(-1000, (10,1,1));  U_bounds[7] = fill(1000, (10,1,1))

# the idx-th training image is used (train set index 3 is a truck at img name index 9, its adversarial couterpart at name index 4 is "deer")
# NOTE! there is no guarantee of finding an optimal solution within the set timelimit below, if an error
# "Result index of attribute MathOptInterface.VariablePrimal(1) out of bounds. There are currently 0 solution(s) in the model."
# is thrown, try a larger timelimit in the function create_CNN_adv
# also, due to the low accuracy, the training image at index 3 might be misclassified already. If this is the case, 
# the index should be changed so that we input a correctly classified image to the create_CNN_adv function.
idx = 3
time, adv = create_CNN_adv(model, idx, "CIFAR10", L_bounds, U_bounds, 600, true, "L1")

# the digit guess of the idx-th training image and the adversarial image
CNN_guess_orig = argmax(model(reshape(x_train[:,:,:,idx], 32, 32, 3, 1)))[1]-1
CNN_guess_adv = argmax(model(reshape(adv, 32, 32, 3, 1)))[1]-1
println("Original training image name index: $CNN_guess_orig, adversarial image name index: $CNN_guess_adv")

@info "Here we display the original image and its adversarial counterpart"

# display the original training image and its adversarial counterpart
convert2image(CIFAR10, x_train[:,:,:,idx])
convert2image(CIFAR10, adv)