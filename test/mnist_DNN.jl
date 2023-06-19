using Flux
using Flux: params, train!, logitcrossentropy, flatten, onehotbatch
using MLDatasets
using Test, Logging
using ML_as_MO

@info "Creating and training a standard ReLU DNN using Flux.jl based on the MNIST digit dataset"
function train_mnist_DNN()
    mnist_DNN = Chain( # 834 nodes
        Dense(784, 24, relu),
        Dense(24, 16, relu),
        Dense(16, 10)
    )
    parameters = params(mnist_DNN)

    x_train, y_train = MNIST(split=:train)[:]
    x_test, y_test = MNIST(split=:test)[:]

    x_train_flatten = flatten(x_train)
    x_test_flatten = flatten(x_test)
    y_train_oh = onehotbatch(y_train, 0:9)
    train = [(x_train_flatten, y_train_oh)]
    test = [(x_test_flatten, y_test)]

    loss(x, y) = logitcrossentropy(mnist_DNN(x), y)

    opt = Adam(0.01) # learning rate of 0.01 gives by far the best results

    println("Value of the loss function at even steps")

    n = 50
    loss_values = zeros(n)
    for i in 1:n
        train!(loss, parameters, train, opt)
        loss_values[i] = loss(x_train_flatten, y_train_oh)
        if i % 10 == 0
            println("Training cycle $i, loss value $(loss_values[i])")
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
    println("Accuracy: $accuracy")

    return mnist_DNN, accuracy
end

mnist_DNN, accuracy = train_mnist_DNN()

# input values bounded to [0,1] (grayscale pixel value)
# other nodes bounded to [-1000,1000] (arbitrary sufficiently large bounds that wont change the output)
U_bounds = Float32[if i <= 784 1 else 1000 end for i in 1:834]
L_bounds = Float32[if i <= 784 0 else -1000 end for i in 1:834]

@info "Testing create_JuMP_model() with bt=\"none\""
time1 = @elapsed begin
    mdl_bt_none = @time create_JuMP_model(mnist_DNN, U_bounds, L_bounds, "none")
end

@info "Testing create_JuMP_model() with bt=\"singlethread\""
time2 = @elapsed begin
    mdl_bt_singlethread = @time create_JuMP_model(mnist_DNN, U_bounds, L_bounds, "singlethread")
end

@info "Testing create_JuMP_model() with bt=\"threads\""
time3 = @elapsed begin
    mdl_bt_threads = @time create_JuMP_model(mnist_DNN, U_bounds, L_bounds, "threads")
end

@info "Testing create_JuMP_model() with bt=\"workers\""
time4 = @elapsed begin
    mdl_bt_workers = @time create_JuMP_model(mnist_DNN, U_bounds, L_bounds, "workers")
end

@info "=============Results=============
Creating the JuMP model without bound tightening (default bounds): $(round(time1; sigdigits = 3))s
Creating the JuMP model with single-threaded bound tightening: $(round(time2; sigdigits = 3))s
Creating the JuMP model with multithreaded bound tightening (Threads): $(round(time3; sigdigits = 3))s
Creating the JuMP model with multithreaded bound tightening (Workers): $(round(time4; sigdigits = 3))s"

@test true

