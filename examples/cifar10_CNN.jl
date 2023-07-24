using Logging
using Statistics
using Flux, Flux.Optimise
using MLDatasets: CIFAR10
using MLDatasets
using Flux: onehotbatch, onecold, flatten, crossentropy, Momentum, logitcrossentropy
using ML_as_MO

include("../src/nn/CNN_JuMP_model.jl")

# This file shows how to use the convert a CNN into a MILP using the create_CNN_model function.
# The CNN training is done similar to the tutorial here: https://fluxml.ai/Flux.jl/stable/tutorials/2020-09-15-deep-learning-flux/

# The CIFAR10 dataset consists of 50 000 32x32x3 images of 10 different classes. The 3rd dimension is the RGB channel.
# The 10 classes are as follows: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

# CNN model using relu activation functions, Flux.flatten and MeanPool pooling layers
# last layer does not use softmax as usual as it cannot be linearized in the MILP
model = Chain( # 4 epochs atm
    Conv((3,3), 3=>5, relu),
    MeanPool((2,2)),
    Conv((3,3), 5=>6, relu),
    MeanPool((2,2)),
    Flux.flatten,
    Dense(216, 24, relu),
    Dense(24, 10),
    # softmax
)

parameters = params(model)

train_x, train_y = CIFAR10(split=:train)[:]
test_x,  test_y  = CIFAR10(split=:test)[:]

train_y_oh = onehotbatch(train_y, 0:9)
test_y_oh = onehotbatch(test_y, 0:9)

train = [(train_x, train_y_oh)]
test = [(test_x, test_y)]

loss(x, y) = sum(logitcrossentropy(model(x), y))
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

opt = Adam(0.01)

n = 50
loss_values = zeros(n)
for i in 1:n
    train!(loss, parameters, train, opt)
    loss_values[i] = loss(train_x, train_y_oh)
    if i % 1 == 0
        println("Training cycle $i, loss value $(loss_values[i])")
    end
end

correct_guesses = 0
test_len = length(test_y)
for i in 1:test_len
    if findmax(model(reshape(test_x[:,:,:,i], 32, 32, 3, 1)))[2][1] - 1 == test_y[i] # -1 to get right index
        correct_guesses += 1
    end
end
acc = correct_guesses / test_len
println("Accuracy: $acc")