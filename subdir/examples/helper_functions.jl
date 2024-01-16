using Flux
using Flux: params, train!, logitcrossentropy, flatten, onehotbatch
using JuMP
using MLDatasets
using ImageShow
using ML_as_MO

# trains a DNN based on the MNIST digit data set
function train_mnist_DNN!(mnist_DNN::Chain)

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

end

# created adversarial images based on a trained MNIST DNN 
function create_adversarials(model::Chain, U_bounds::Vector{Float32}, L_bounds::Vector{Float32}, count::Int64, l_norm::String="L1")
    K = length(model)
    x_train, y_train = MNIST(split=:train)[:]
    x_train_flatten = flatten(x_train)

    adversarial_images = []
    times = []
    for i in 1:count
        println("Generating adversarial image with $l_norm-norm, index $i")
        false_class = create_JuMP_model(model, U_bounds, L_bounds, "none", true)
        cur_digit = y_train[i]
        cur_digit_img = x_train_flatten[:, i]

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

function show_digit(img_flatten)
    img = reshape(img_flatten, 28, 28)
    plot = convert2image(MNIST, img)
    return plot
end

# created adversarial images (L1 or L2-norm) based on a trained MNIST CNN
function create_CNN_adv(model::Chain, idx::Int64, CNN_data::String, L_bounds::Vector{Array{Float32}}, U_bounds::Vector{Array{Float32}}, time_limit::Int64=600, verbose::Bool=false, l_norm::String="L1")
    @assert l_norm == "L1" || l_norm == "L2" "l_norm must be either \"L1\" or \"L2\""
    @assert CNN_data == "MNIST" || CNN_data == "CIFAR10" "CNN_data must be either \"MNIST\" or \"CIFAR10\""

    if CNN_data == "MNIST"

        K = length(model)
        x_train, y_train = MNIST(split=:train)[:]

        false_class = create_CNN_JuMP_model(model, (28,28,1,1), L_bounds, U_bounds)
        set_optimizer_attribute(false_class, "TimeLimit", time_limit)
        set_optimizer_attribute(false_class, "OutputFlag", verbose)
        cur_digit = y_train[idx]
        cur_digit_img = x_train[:, :, 1, idx]

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

        if l_norm == "L1"
            @objective(false_class, Min, sum(d[0, 1, h, w] for h in 1:28, w in 1:28))
        elseif l_norm == "L2"
            @objective(false_class, Min, sum(d[0, 1, h, w]^2 for h in 1:28, w in 1:28))
        end

        time = @elapsed optimize!(false_class)

        adversarial = zeros(Float32, 28, 28)
        for h in 1:28
            for w in 1:28
                pixel_value = value(false_class[:x][0, 1, h, w]) # indexing
                adversarial[h, w] = pixel_value
            end
        end

        return time, adversarial

    elseif CNN_data == "CIFAR10"

        K = length(model)
        x_train, y_train = CIFAR10(split=:train)[:]

        false_class = create_CNN_JuMP_model(model, (32,32,3,1), L_bounds, U_bounds)
        set_optimizer_attribute(false_class, "TimeLimit", time_limit)
        set_optimizer_attribute(false_class, "OutputFlag", verbose)
        cur_img_name = y_train[idx]
        cur_img = x_train[:, :, :, idx]

        x = false_class[:x]
        @variable(false_class, d[k in [0], i in 1:3, h in 1:32, w in 1:32] >= 0)
        mult = 1.2
        imposed_index = (cur_img_name + 5) % 10 + 1

        # adversarial output index must have largest activation
        for output_node in 1:10
            if output_node != imposed_index
                @constraint(false_class, x[K-1, imposed_index, 1, 1] >= mult * x[K-1, output_node, 1, 1])
            end
        end

        # d variable bounds input nodes
        for i in 1:3
            for h in 1:32
                for w in 1:32
                    @constraint(false_class, -d[0, i, h, w] <= x[0, i, h, w] - cur_img[h,w,i])
                    @constraint(false_class, x[0, i, h, w] - cur_img[h,w,i] <= d[0, i, h, w])
                end
            end
        end

        if l_norm == "L1"
            @objective(false_class, Min, sum(d[0, i, h, w] for i in 1:3, h in 1:32, w in 1:32))
        elseif l_norm == "L2"
            @objective(false_class, Min, sum(d[0, i, h, w]^2 for i in 1:3, h in 1:32, w in 1:32))
        end

        time = @elapsed optimize!(false_class)

        adversarial = zeros(Float32, 32, 32, 3)
        for i in 1:3
            for h in 1:32
                for w in 1:32
                    pixel_value = value(false_class[:x][0, i, h, w]) # indexing
                    adversarial[h, w, i] = pixel_value
                end
            end
        end

        return time, adversarial

    end
end

# extracts the input layer pixel values from an optimised CNN MILP model
function extract_CNN_input(CNN_model::Model, input_size::Tuple{Int64, Int64, Int64})
    x = CNN_model[:x]
    input = zeros(Float32, input_size[1], input_size[2], input_size[3])
    for i in input_size[1]
        for h in 1:input_size[2]
            for w in 1:input_size[3]
                input[i,h,w] = value(x[0,i,h,w])
            end
        end
    end
    return input
end