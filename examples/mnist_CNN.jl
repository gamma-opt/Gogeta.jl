using Logging
using Statistics
using Flux
using Flux: onehotbatch, flatten, logitcrossentropy
using MLDatasets: MNIST
using MLDatasets
using ML_as_MO

include("../src/nn/CNN_JuMP_model.jl")
include("helper_functions.jl")

@info "Creating and training a small ReLU CNN using Flux.jl based on the MNIST digit dataset
       The CNN is trained for 50 training cycles with the full training data set"

x_train, y_train = MNIST(split=:train)[:]
x_test, y_test = MNIST(split=:test)[:]

x_train = reshape(x_train, 28, 28, 1, 60000)
x_test = reshape(x_test, 28, 28, 1, 10000)

y_train_oh = onehotbatch(y_train, 0:9)

train = [(x_train, y_train_oh)]
test = [(x_test, y_test)]

model = Chain(
    Conv((5,5), 1=>4, relu),
    MeanPool((2,2)),
    flatten,
    Dense(576, 16, relu),
    Dense(16, 10),
)

p = Flux.params(model)
loss(x, y) = logitcrossentropy(model(x), y)
opt = Adam(0.01)

# training the CNN
n = 50
loss_values = zeros(n)
for i in 1:n
    train!(loss, p, train, opt)
    loss_values[i] = loss(x_train, y_train_oh)
    if i % 5 == 0
        println("Training cycle $i, loss value $(loss_values[i])")
    end
end

# calculating the accuracy of the CNN
correct_guesses = 0
test_len = length(y_test)
for i in 1:test_len
    if findmax(model(reshape(x_test[:,:,:,i], 28, 28, 1, 1)))[2][1] - 1  == y_test[i] # -1 to get right index
        correct_guesses += 1
    end
end
acc = correct_guesses / test_len
println("Accuracy of the CNN: $(acc)%")

@info "Creating a MILP model and generating one adversarial image based on the trained ReLU CNN
       A timelimit of 600 sec is used in the adversarial image optimisation problem"

# the index-th training image is used
idx = 1
time, adv = create_CNN_adv(model, idx, "image", 600, true)

# the digit guess of the index-th training image and the adversarial image
CNN_guess_original = argmax(model(reshape(x_train[:,:,:,idx], 28, 28, 1, 1)))[1]-1
CNN_guess_adv = argmax(model(reshape(adv, 28, 28, 1, 1)))[1]-1
println("Original training image guess: $CNN_guess_original, adversarial image guess: $CNN_guess_adv")

# display the original training image and its adversarial counterpart
convert2image(MNIST, reshape(x_train[:,:,:,idx], 28, 28))
convert2image(MNIST, reshape(adv, 28, 28))
