# using Logging
# using Statistics
# using Flux, Flux.Optimise
# using MLDatasets: CIFAR10
# using MLDatasets
# using Flux: onehotbatch, onecold, flatten, crossentropy, Momentum, logitcrossentropy
# using ML_as_MO

# include("../src/nn/CNN_JuMP_model.jl")

# # This file shows how to use the convert a CNN into a MILP using the create_CNN_model function.
# # The CNN training is done similar to the tutorial here: https://fluxml.ai/Flux.jl/stable/tutorials/2020-09-15-deep-learning-flux/

# # The CIFAR10 dataset consists of 50 000 32x32x3 images of 10 different classes. The 3rd dimension is the RGB channel.
# # The 10 classes are as follows: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.




# # CNN model using relu activation functions, Flux.flatten and MeanPool pooling layers
# # last layer does not use softmax as usual as it cannot be linearized in the MILP
# model = Chain( # 4 epochs atm
#     Conv((5,5), 3=>5, relu),
#     MeanPool((2,2)),
#     Conv((5,5), 5=>6, relu),
#     MeanPool((2,2)),
#     Flux.flatten,
#     Dense(150, 16, relu),
#     Dense(16, 10),
#     # softmax
# )

# parameters = params(model)

# train_x, train_y = CIFAR10(split=:train)[:]
# test_x,  test_y  = CIFAR10(split=:test)[:]

# # train_x_flatten = flatten(train_x)
# # test_x_flatten = flatten(test_x)
# train_y_oh = onehotbatch(train_y, 0:9)
# test_y_oh = onehotbatch(test_y, 0:9)

# train = [(train_x, train_y_oh)]
# test = [(test_x, test_y)]

# # using Plots
# # using Images.ImageCore

# # image(x) = colorview(RGB, permutedims(x, (3, 2, 1)))
# # plot(image(train_x[:,:,:,1]))

# loss(x, y) = sum(logitcrossentropy(model(x), y))
# accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

# opt = Adam(0.01)

# n = 50
# loss_values = zeros(n)
# for i in 1:n
#     train!(loss, parameters, train, opt)
#     loss_values[i] = loss(train_x, train_y_oh)
#     if i % 1 == 0
#         println("Training cycle $i, loss value $(loss_values[i])")
#     end
# end

# correct_guesses = 0
# test_len = length(test_y)
# for i in 1:test_len
#     if findmax(model(reshape(test_x[:,:,:,i], 32, 32, 3, 1)))[2][1] - 1 == test_y[i] # -1 to get right index
#         correct_guesses += 1
#     end
# end
# acc = correct_guesses / test_len
# println("Accuracy: $acc")






using Logging
using Statistics
using Flux, Flux.Optimise
using MLDatasets: MNIST
using MLDatasets
using Flux: onehotbatch, onecold, flatten, crossentropy, logitcrossentropy
using ML_as_MO

include("../src/nn/CNN_JuMP_model.jl")

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
    Flux.flatten,
    Dense(576, 16, relu),
    Dense(16, 10),
)

p = Flux.params(model)

loss(x, y) = logitcrossentropy(model(x), y)

opt = Adam(0.01)

n = 50
loss_values = zeros(n)
for i in 1:n
    train!(loss, p, train, opt)
    loss_values[i] = loss(x_train, y_train_oh)
    if i % 1 == 0
        println("Training cycle $i, loss value $(loss_values[i])")
    end
end

correct_guesses = 0
test_len = length(y_test)
for i in 1:test_len
    if findmax(model(reshape(x_test[:,:,:,i], 28, 28, 1, 1)))[2][1] - 1  == y_test[i] # -1 to get right index
        correct_guesses += 1
    end
end
acc = correct_guesses / test_len

JuMP_model = create_CNN_model(model, (28,28,1,1), true)

function create_adv_CNN(model::Chain, i::Int64)
    K = length(model)
    x_train, y_train = MNIST(split=:train)[:]
    # adversarial_images = []
    # times = []

    false_class = create_CNN_model(model, (28,28,1,1), true)
    cur_digit = y_train[i]
    cur_digit_img = x_train[:, :, 1, i]

    x = false_class[:x]
    @variable(false_class, d[k in [0], i in [1], h in 1:28, w in 1:28] >= 0)
    mult = 1.2
    imposed_index = (cur_digit + 5) % 10 + 1

    # adversarial output index must have largest activation
    for output_node in 1:10
        if output_node != imposed_index
            @constraint(false_class, x[K-1, imposed_index, 1, 1] >= mult * x[K-1, output_node, 1, 1])
        end
    end

    # d variable bounds input nodes
    for h in 1:28
        for w in 1:28
            @constraint(false_class, -d[0, 1, h, w] <= x[0, 1, h, w] - cur_digit_img[h,w])
            @constraint(false_class, x[0, 1, h, w] - cur_digit_img[h,w] <= d[0, 1, h, w])
        end
    end

    #
    @objective(false_class, Min, sum(d[0, 1, h, w] for h in 1:28, w in 1:28))

    time = @elapsed optimize!(false_class)

    adversarial = zeros(Float32, 28, 28)
    for h in 1:28
        for w in 1:28
            pixel_value = value(false_class[:x][0, 1, h, w]) # indexing
            adversarial[h, w] = pixel_value
        end
    end

    return time, adversarial
end

time, adv = create_adv_CNN(model, 1)

x = JuMP_model[:x]

model(reshape(x_test[:,:,:,2], 28, 28, 1, 1))
convert2image(MNIST, reshape(x_test[:,:,:,2], 28, 28))


function extract_input(CNN_model::Model, input_size)
    x = CNN_model[:x]
    input = zeros(Float32, input_size[1], input_size[2])
    for h in 1:input_size[1]
        for w in 1:input_size[2]
            input[h,w] = value(x[0,1,h,w])
        end
    end
    return input
end

@objective(JuMP_model, Max, x[4,1,1,1])
optimize!(JuMP_model)
result0 = extract_input(JuMP_model, (28,28))
