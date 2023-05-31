# NOTE! This file inclused some functions for training a MNIST digit nn,
# viewing the output nodes of the nn, viewing 28x28 grayscale images,
# creating adversarial images and a confusion matrix for a train and test set

# trains a mnist digit nn and returns it, (only) train and test data as input
function train_mnist_nn(mnist_nn, seed = abs(rand(Int)))
    Random.seed!(seed) # seed for reproducibility

    # mnist_nn = Chain(
    #     Dense(784, 32, relu),
    #     Dense(32, 16, relu),
    #     Dense(16, 10)
    # )
    x_train, y_train = MNIST(split=:train)[:]
    x_test, y_test = MNIST(split=:test)[:]

    parameters = params(mnist_nn)
    x_train_flatten = flatten(x_train)
    x_test_flatten = flatten(x_test)
    y_train_oh = onehotbatch(y_train, 0:9)
    train = [(x_train_flatten, y_train_oh)]
    test = [(x_test_flatten, y_test)]

    loss(x, y) = Flux.Losses.logitcrossentropy(mnist_nn(x), y)

    opt = ADAM(0.01) # learning rate of 0.01 gives by far the best results

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
        if findmax(mnist_nn(test[1][1][:, i]))[2] - 1  == y_test[i] # -1 to get right index
            correct_guesses += 1
        end
    end
    accuracy = correct_guesses / test_len
    println("Accuracy: ", accuracy)

    return mnist_nn, accuracy
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

# prints last layer acivations for a flattened image (784 pixel array)
function digit_activations(model, img_flatten)
    guess_digit = create_JuMP_model(model, "predict")
    evaluate(guess_digit, img_flatten)
    optimize!(guess_digit)
    nn_activations = model(img_flatten)
    
    return nn_activations
end

# shows a flattened gray scale image 
function show_digit(img_flatten)
    img = reshape(img_flatten, 28, 28)
    plot = convert2image(MNIST, img)
    return plot
end

# creates adversarial images from th MNIST train data 
function create_adversarials(model, start_idx, end_idx, l_norm = "L1")

    @assert l_norm == "L1" || l_norm == "L2" "l_norm must be \"L1\" or \"L2\""
    adversarial_images = []
    for i in start_idx:end_idx
        println("L-norm: ", l_norm, ", Index: ", i)
        false_class = create_JuMP_model(model, "missclassified $l_norm", i)
        optimize!(false_class)

        adversarial = Float32[]
        for pixel in 1:784
            pixel_value = value(false_class[:x][0, pixel]) # indexing
            push!(adversarial, pixel_value)
        end

        push!(adversarial_images, adversarial)
    end

    return adversarial_images
end

# adversarial_L1 = create_adversarials(mnist_model,1,12000, "L1") # Note! very long runtime
# adversarial_L2 = create_adversarials(mnist_model,1,12000, "L2")

# creates a confusion matrix for MNIST digits
function confusion_matrix(model, features, labels)
    # ordering = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    features_flatten = flatten(features)
    len = length(labels)
    predictions = zeros(len)
    for i in 1:len
        cur_pred = findmax(model(features_flatten[:,i]))[2] - 1
        predictions[i] = cur_pred
    end
    return ConfusionMatrix()(predictions, labels)
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

# function that creates a label to the digit image
function create_labeled_digit(original_img, label)
    img = load(original_img)
    plot(img, framestyle=:none, title="Label is $label")
    savefig("uncropped_$label.png")

    img_label = load("uncropped_$label.png")
    # img_size = size(img_label)
    # cropping to 400x400 with even edges
    cropped = img_label[:, 116:515] 
    # img_smaller = imresize(cropped, (200, 200))
    return cropped
end
