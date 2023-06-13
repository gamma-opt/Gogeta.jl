# this file contains an example of adversarial image generation using the package.
# the package is used to generate 10 adversarial images based on the MNIST train set

using MLDatasets
using Flux
using Flux: params, train!, mse, flatten, onehotbatch
using JuMP
using JuMP: Model, value
using Gurobi
using Random

Random.seed!(42)

# create the DNN used as a reference in the 0-1 MILP surrogate model
mnist_DNN = Chain(
    Dense(784, 32, relu),
    Dense(32, 16, relu),
    Dense(16, 10)
)
K = length(mnist_DNN)

# train the mnist_DNN using the MNIST digit data set
x_train, y_train = MNIST(split=:train)[:]
x_test, y_test = MNIST(split=:test)[:]

x_train_flatten = flatten(x_train)
x_test_flatten = flatten(x_test)
y_train_oh = onehotbatch(y_train, 0:9)
train = [(x_train_flatten, y_train_oh)]
test = [(x_test_flatten, y_test)]

# DNN training paraemeters
parameters = params(mnist_DNN)
loss(x, y) = Flux.Losses.logitcrossentropy(mnist_DNN(x), y)
opt = ADAM(0.01)
n = 50

# training the DNN for n epochs
println("Value of the loss function at even steps")
loss_values = zeros(n)
for i in 1:n
    train!(loss, parameters, train, opt)
    loss_values[i] = loss(x_train_flatten, y_train_oh)
    if i % 10 == 0
        println("Epoch $i, loss $(loss_values[i])")
    end
end

# prints the accuracy on the test set
correct_guesses = 0
test_len = length(y_test)
for i in 1:test_len
    if findmax(mnist_DNN(test[1][1][:, i]))[2] - 1  == y_test[i] # -1 to get right index
        correct_guesses += 1
    end
end
accuracy = correct_guesses / test_len
println("Training complete, DNN accuracy $(accuracy)%")

# initial lower and upper bounds for each node in the DNN
# input layer (nodes 1:784) is bounded to [0,1] to correspond with a grayscale pixel value
# other layers (nodes 785:842) are bounded to [-1000, 1000] (arbitrary sufficiently large bounds)
initial_U_bounds = Float32[if i <= 784 1 else 1000 end for i in 1:842]
initial_L_bounds = Float32[if i <= 784 0 else -1000 end for i in 1:842]

# converts the mnist_DNN to a JuMP model formualtion 
JuMP_model = create_JuMP_model(mnist_DNN, initial_U_bounds, initial_L_bounds, false)

# take an arbitrary digit image from the data set
i = 1
cur_digit = y_train[i]
cur_digit_img = x_train_flatten[:, i]

# store the "x" variable from the JuMP_model
x = JuMP_model[:x]

# new variables to capture the L1-norm difference between original and adversarial image
@variable(JuMP_model, d[k in [0], j in 1:784] >= 0)

# value of imposed digit must be higer than in other output nodes
mult = 1.2
imposed_index = (cur_digit + 5) % 10 + 1 # digit d is imposed as (d + 5 mod 10) (+1 for indexing)
for output_node in 1:10
    if output_node != imposed_index
        @constraint(JuMP_model, x[K, imposed_index] >= mult * x[K, output_node])
    end
end

# d variables bound the differences in pixel values between the two images
for input_node in 1:784
    @constraint(JuMP_model, -d[0, input_node] <= x[0, input_node] - cur_digit_img[input_node])
    @constraint(JuMP_model, x[0, input_node] - cur_digit_img[input_node] <= d[0, input_node])
end

# Objective: minimize the difference between pixel values 
@objective(JuMP_model, Min, sum(d[0, input_node] for input_node in 1:784))

@time optimize!(JuMP_model)

# construct the adversarial image from the input layer variables
adversarial = zeros(Float32, 28, 28)
for pixel in 1:784
    pixel_value = value(JuMP_model[:x][0, pixel]) # indexing
    adversarial[pixel] = pixel_value
end

# plot the adversarial image
convert2image(MNIST, adversarial)
