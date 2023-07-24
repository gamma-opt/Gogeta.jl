using Logging
using Statistics
using Flux
using Flux: onehotbatch, flatten, logitcrossentropy, train!
using MLDatasets
using MLDatasets: CIFAR10
using ML_as_MO

include("../src/nn/CNN_JuMP_model.jl")
include("helper_functions.jl")

# This file shows how to use the convert a CNN into a MILP using the create_CNN_model function.

# The CIFAR10 dataset consists of 50 000 32x32x3 images of 10 different classes. The 3rd dimension is the RGB channel.
# The 10 classes are as follows: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

@info "Creating and training a small ReLU CNN using Flux.jl based on the CIFAR10 image dataset
       The CNN is trained for 50 training cycles with the full training data set"

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

# training the CNN
n = 50
loss_values = zeros(n)
for i in 1:n
    train!(loss, parameters, train, opt)
    loss_values[i] = loss(x_train, y_train_oh)
    if i % 1 == 0
        println("Training cycle $i, loss value $(loss_values[i])")
    end
end

# calculating the accuracy of the CNN (around 35% with these parameters)
correct_guesses = 0
test_len = length(test_y)
for i in 1:test_len
    if findmax(model(reshape(test_x[:,:,:,i], 32, 32, 3, 1)))[2][1] - 1 == test_y[i] # -1 to get right index
        correct_guesses += 1
    end
end
acc = correct_guesses / test_len
println("Accuracy: $acc")

# the index-th training image is used (index 3 is a truck at img name index 9, its couterpart at index 4 is "deer")
idx = 3
time, adv = create_CNN_adv(model, idx, "CIFAR10", 600, true, "L1")

# the digit guess of the index-th training image and the adversarial image
CNN_guess_orig = argmax(model(reshape(x_train[:,:,:,idx], 32, 32, 3, 1)))[1]-1
CNN_guess_adv = argmax(model(reshape(adv, 32, 32, 3, 1)))[1]-1
println("Original training image name index: $CNN_guess_orig, adversarial image name index: $CNN_guess_adv")

# display the original training image and its adversarial counterpart
convert2image(CIFAR10, x_train[:,:,:,idx])
convert2image(CIFAR10, adv)