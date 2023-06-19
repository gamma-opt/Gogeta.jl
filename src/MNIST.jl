# NOTE! This file inclused some functions for training a MNIST digit nn,
# viewing the output nodes of the nn, viewing 28x28 grayscale images,
# creating adversarial images and a confusion matrix for a train and test set

# trains a mnist digit nn and returns it, (only) train and test data as input
function train_mnist_nn(mnist_DNN::Chain, seed::Int64=abs(rand(Int)))
    Random.seed!(seed) # seed for reproducibility

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

# finds an optimal input to maximize a given output node
function create_optimal_input(model, digit)
    max_activation = create_JuMP_model(model, "predict", -1, digit)
    optimize!(max_activation)

    optimal_img_flatten = Float32[]
    for pixel in 1:784
        pixel_value = value(max_activation[:x][0, pixel]) # indexing
        append!(optimal_img_flatten, pixel_value)
    end

    return optimal_img_flatten
end

# shows a flattened gray scale image 
function show_digit(img_flatten)
    img = reshape(img_flatten, 28, 28)
    plot = convert2image(MNIST, img)
    return plot
end

# creates adversarial images from th MNIST train data 
function create_adversarials(model::Chain, U::Vector{Float32}, L::Vector{Float32}, start_idx, end_idx, l_norm = "L1", BT::Bool=false)
    K = length(model)
    x_train, y_train = MNIST(split=:train)[:]
    x_train_flatten = flatten(x_train)

    @assert l_norm == "L1" || l_norm == "L2" "l_norm must be \"L1\" or \"L2\""
    adversarial_images = []
    times = []
    for i in start_idx:end_idx
        cur_digit = y_train[i]
        cur_digit_img = x_train_flatten[:, i]

        println("L-norm: ", l_norm, ", Index: ", i)
        false_class = create_JuMP_model(model, U, L, BT)
        x = false_class[:x]
        # variables for constraint (12) in the 2018 paper
        @variable(false_class, d[k in [0], j in 1:784] >= 0)
        mult = 1.2
        imposed_index = (cur_digit + 5) % 10 + 1 # digit is imposed as (d + 5 mod 10), +1 for indexing
        for output_node in 1:10
            if output_node != imposed_index
                @constraint(false_class, x[K, imposed_index] >= mult * x[K, output_node])
            end
        end
        for input_node in 1:784
            @constraint(false_class, -d[0, input_node] <= x[0, input_node] - cur_digit_img[input_node])
            @constraint(false_class, x[0, input_node] - cur_digit_img[input_node] <= d[0, input_node])
        end
        if l_norm == "L1"
            @objective(false_class, Min, sum(d[0, input_node] for input_node in 1:784))
        else 
            @objective(false_class, Min, sum(d[0, input_node]^2 for input_node in 1:784))
        end

        time = @elapsed optimize!(false_class)

        adversarial = Float32[]
        for pixel in 1:784
            pixel_value = value(false_class[:x][0, pixel]) # indexing
            push!(adversarial, pixel_value)
        end

        push!(adversarial_images, adversarial)
        push!(times, time)
    end

    return times, adversarial_images
end

# adds a certain amount of adversarials
function add_adversarials(adversarials, count)
    @assert 1 <= count <= 12000 "count must be between 1 and 12000 (max 20%)"
    x_train, y_train = MNIST(split=:train)[:]

    # creating correct sized adv features and labels
    adversarial_features = adversarials[1:count] 
    adversarial_labels = y_train[1:count]

    reshaped = map(x -> reshape(x, 28, 28), adversarial_features)
    reshaped2matrix = Array{Float32, 3}(undef, 28, 28, count)
    for i in 1:count
        reshaped2matrix[:,:,i] = reshaped[i]
    end

    new_x_train = cat(x_train, reshaped2matrix; dims=3)
    new_y_train = cat(y_train, adversarial_labels; dims=1)

    return new_x_train, new_y_train
end