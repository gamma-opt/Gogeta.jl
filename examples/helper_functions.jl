using Flux
using Flux: params, train!, logitcrossentropy, flatten, onehotbatch
using JuMP
using MLDatasets
using ImageShow
using ML_as_MO

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



# using JuMP
# using Gurobi

# m = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0))

# @variable(m, x[1:5] >= 0)

# for i in 1:5
#     name = "constraint $i"
#     @constraint(m, "constraint $i", x[i] <= 2*i)
# end

# println(m)

# delete(m, first)
# unregister(m, :first)

# println(m)

# @objective(m, Max, 2*y + x)

# optimize!(m)
# objective_value(m)

# @variable(model, x[k in 0:K, j in 1:node_count[k+1]] >= 0)