using Flux
using Flux: params, train!, mse, flatten, onehotbatch
using MLDatasets
using Test, Logging
using ML_as_MO

function train_mnist_DNN(mnist_DNN::Chain, seed::Int64=abs(rand(Int)))
    # Random.seed!(seed) # seed for reproducibility

    x_train, y_train = MNIST(split=:train)[:]
    x_test, y_test = MNIST(split=:test)[:]

    parameters = params(mnist_DNN)
    x_train_flatten = flatten(x_train)
    x_test_flatten = flatten(x_test)
    y_train_oh = onehotbatch(y_train, 0:9)
    train = [(x_train_flatten, y_train_oh)]
    test = [(x_test_flatten, y_test)]

    loss(x, y) = Flux.Losses.logitcrossentropy(mnist_DNN(x), y)

    opt = Adam(0.01) # learning rate of 0.01 gives by far the best results

    println("Value of the loss function at even steps")

    n = 50
    loss_values = zeros(n)
    for i in 1:n
        train!(loss, parameters, train, opt)
        loss_values[i] = loss(x_train_flatten, y_train_oh)
        if i % 10 == 0
            println(loss_values[i])
        end
    end

    correct_guesses = 0
    test_len = length(y_test)
    for i in 1:test_len
        if findmax(mnist_DNN(test[1][1][:, i]))[2] - 1  == y_test[i] # -1 to get right index
            correct_guesses += 1
        end
    end
    accuracy = correct_guesses / test_len
    println("Accuracy: ", accuracy)

    return mnist_DNN, accuracy
end

@info "creating and training a ReLU DNN based on MNIST digit dataset"
mnist_DNN = Chain( # 834 nodes
    Dense(784, 24, relu),
    Dense(24, 16, relu),
    Dense(16, 10)
)
mnist_DNN, acc = train_mnist_DNN(mnist_DNN)

U_bounds = Float32[if i <= 784 1 else 1000 end for i in 1:834]
L_bounds = Float32[if i <= 784 0 else -1000 end for i in 1:834]

@info "Testing create_JuMP_model() with bt=\"none\""
@time mdl = create_JuMP_model(mnist_DNN, U_bounds, L_bounds, "none")

@info "Testing create_JuMP_model() with bt=\"singlethread\""
@time mdl_bt_singlethread = create_JuMP_model(mnist_DNN, U_bounds, L_bounds, "singlethread")

@info "Testing create_JuMP_model() with bt=\"threads\""
@time mdl_bt_threads = create_JuMP_model(mnist_DNN, U_bounds, L_bounds, "threads")

@info "Testing create_JuMP_model() with bt=\"workers\""
@time mdl_bt_workers = create_JuMP_model(mnist_DNN, U_bounds, L_bounds, "workers")

@test true

# using Test
# using ML_as_MO

# @test joo(1) == 2
# @test joo(2) != 1