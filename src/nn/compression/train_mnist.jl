using Flux
using Flux: crossentropy, params, onehotbatch, onecold, throttle, train!, logitcrossentropy
using Flux.Data: DataLoader
using MLDatasets
using Statistics

include("./lossless_compression.jl")
include("../bound_tightening.jl")

const PATIENCE = 3  # Number of epochs to wait before stopping when accuracy does not improve
const CHECK_EVERY = 5  # Check the test accuracy every 5 epochs

function downsample_image(img::Array{T,2}) where T
  out_dim = size(img, 1) ÷ 2
  out = zeros(T, out_dim, out_dim)
  for i in 1:2:size(img, 1)-1
      for j in 1:2:size(img, 2)-1
          out[div(i+1,2), div(j+1,2)] = (img[i,j] + img[i+1,j] + img[i,j+1] + img[i+1,j+1]) / 4
      end
  end
  return out
end

kaiming_initialization(out, inp) = sqrt(2.0 / inp) * randn(out, inp)

function train_model(n_neurons_hidden, optimizer_params, regularization_params, downsize_images = true)
  x_train, Y_train = MNIST(split=:train)[:]
  x_test, Y_test = MNIST(split=:test)[:]

  if downsize_images
    x_train = mapslices(downsample_image, x_train; dims=(1,2))
    x_test = mapslices(downsample_image, x_test; dims=(1,2))
  end

  x_train, x_test = Flux.flatten(x_train), Flux.flatten(x_test)
  Y_train, Y_test = onehotbatch(Y_train, 0:9), onehotbatch(Y_test, 0:9)

  batchsize = 64
  train_loader = DataLoader((x_train, Y_train), batchsize=batchsize, shuffle=true)
  test_loader = DataLoader((x_test, Y_test), batchsize=batchsize)

  n_input_layer = downsize_images ? 14^2 : 28^2
  n_neurons = [n_input_layer, n_neurons_hidden..., 10]
  layers = []
  for i = 1:length(n_neurons)-2
    # Initialize using Kaiming initialization for weights and zeros for biases
    W_init = kaiming_initialization(n_neurons[i+1], n_neurons[i])
    b_init = zeros(n_neurons[i+1])

    layer = Dense(W_init, b_init, relu)
    push!(layers, layer)
  end

  W_init = kaiming_initialization(n_neurons[end], n_neurons[end-1])
  b_init = zeros(n_neurons[end])
  push!(layers, Dense(W_init, b_init))

  model = Chain(layers...)

  optimizer_index, o1, o2, o3 = optimizer_params
  if optimizer_index == 1
    optimizer = Momentum(o1, o2)
  elseif optimizer_index == 2
    optimizer = ADAM(o1, (o2, o3))
  elseif optimizer_index == 3
    optimizer = Descent(o1)
  end

  l1norm(x) = sum(abs, x)
  l2norm(x) = sum(norm, x)

  function create_loss_function(regularization_index, r1)
    if regularization_index == 1
      # L1 regularization
      return (x, y) -> logitcrossentropy(model(x), y) + r1 * sum(l1norm, Flux.params(model))
    elseif regularization_index == 2
      # L2 regularization
      return (x, y) -> logitcrossentropy(model(x), y) + r1 * sum(l2norm, Flux.params(model))
    else
      # No regularization
      return (x, y) -> logitcrossentropy(model(x), y)
    end
  end

  r_index, r1 = regularization_params
  loss = create_loss_function(r_index, r1)
  prms = params(model)

  loss_values = Float32[]

  previous_test_acc = 0.0
  non_improving_epochs = 0

  epoch = 1

  while true
      batch_loss_values = zeros(length(train_loader))
      for (i, (x, y)) in enumerate(train_loader)
          train!(loss, prms, [(x, y)], optimizer)
          batch_loss_values[i] = loss(x, y)
      end
      push!(loss_values, mean(batch_loss_values))

      if epoch % CHECK_EVERY == 0
          accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
          test_acc = mean([accuracy(x_batch, y_batch) for (x_batch, y_batch) in test_loader])
          test_loss = mean([loss(x_batch, y_batch) for (x_batch, y_batch) in test_loader])

          println("Epoch: $epoch, Test Loss: $test_loss, Test Accuracy: $test_acc")

          # Exit the loop if test accuracy does not improve for PATIENCE epochs
          if test_acc <= previous_test_acc
              non_improving_epochs += 1
              if non_improving_epochs >= PATIENCE
                  break
              end
          else
              non_improving_epochs = 0
              previous_test_acc = test_acc
          end
      end

      # Decay the learning rate every 50 epochs
      if epoch % 50 == 0
          optimizer.eta *= 0.1
      end

      epoch += 1
  end

  println("Training finished after $epoch epochs")

  # round the weights smaller than 1e-5 to 0
  for p in params(model)
    p[abs.(p) .< 1e-3] .= 0
  end

  return model, test_loader, n_neurons
end

function perform_tests(test_cases)
  results = []

  for (i, test_case) in enumerate(test_cases)
    # [hidden_layers], (1: Momentum, 2: ADAM, 3: SGD, lr, momentum), (1: L1, 2: L2, 3: None, regularization strength), downsize_images
    model_params, optimizer_params, regularization_params, downsize_images = test_case

    println("$i: Test with $model_params hidden layers")

    model, test_loader, n_neurons = train_model(model_params, optimizer_params, regularization_params, downsize_images)

    n_zero_weights = sum(p -> sum(abs.(p) .< 1e-10), params(model))
    println("Number of zero weights: ($n_zero_weights / $(sum(length, params(model))))")

    n_neurons_initial = sum(n_neurons)
    U_bounds = Float32[if i <= n_neurons[1] 1 else 10000 end for i in 1:n_neurons_initial]
    L_bounds = Float32[if i <= n_neurons[1] 0 else -10000 end for i in 1:n_neurons_initial]

    best_U_bounds, best_L_bounds = bound_tightening(model, U_bounds, L_bounds)

    best_U_bounds = best_U_bounds[n_neurons[1] + 1:end]
    best_L_bounds = best_L_bounds[n_neurons[1] + 1:end]

    W1, b1, W2, b2, W3, b3 = params(model)

    upper1 = best_U_bounds[1:n_neurons[2]]
    upper2 = best_U_bounds[n_neurons[2] + 1:n_neurons[2] + n_neurons[3]]
    upper3 = best_U_bounds[n_neurons[2] + n_neurons[3] + 1:end]

    lower1 = best_L_bounds[1:n_neurons[2]]
    lower2 = best_L_bounds[n_neurons[2] + 1:n_neurons[2] + n_neurons[3]]
    lower3 = best_L_bounds[n_neurons[2] + n_neurons[3] + 1:end]

     # TODO: update these to work with more layers
    W1_pruned, b1_pruned, W2_pruned, b2_pruned = prune_from_model(W1, b1, W2, b2, upper1, upper2, lower1, lower2)
    W2_pruned2, b2_pruned2, W3_pruned, b3_pruned = prune_from_model(W2_pruned, b2_pruned, W3, b3, upper2, upper3, lower2, lower3)
    layer_parameters = [(W1_pruned, b1_pruned), (W2_pruned2, b2_pruned2), (W3_pruned, b3_pruned)]

    n_neurons_pruned = [size(layer_parameters[1][1], 2), map(x -> size(x[1], 1), layer_parameters)...]

    layers = []
    for i = 1:length(n_neurons_pruned)-2
      layer = Dense(n_neurons_pruned[i], n_neurons_pruned[i+1], relu)
      push!(layers, layer)
    end
    push!(layers, Dense(n_neurons_pruned[end-1], n_neurons_pruned[end]))

    model_pruned = Chain(layers...)

    for (i, (w, b)) in enumerate(layer_parameters)
      model_pruned.layers[i].weight .= w
      model_pruned.layers[i].bias .= b
    end

    num_neurons_original = sum(length, params(model))
    num_neurons_pruned = sum(length, params(model_pruned))
    # println("Number of neurons pruned: $(num_neurons_original - num_neurons_pruned) (a $(round((num_neurons_original - num_neurons_pruned) / num_neurons_original * 100, digits=2))% reduction)")

    accuracy(x, y, m) = mean(onecold(m(x)) .== onecold(y))
    test_acc_original = mean([accuracy(x_batch, y_batch, model) for (x_batch, y_batch) in test_loader])
    test_acc_pruned = mean([accuracy(x_batch, y_batch, model_pruned) for (x_batch, y_batch) in test_loader])

    if (round(test_acc_original, digits=4) != round(test_acc_pruned, digits=4))
      push!(results, (test_acc_pruned, num_neurons_pruned, num_neurons_original, false))
    end

    push!(results, (test_acc_pruned, num_neurons_pruned, num_neurons_original, true))
  end

  return results
end

test_cases = [
  [[100, 25], (1, 0.01, 0.9, 0), (1, 0.0005), true]
]

results = perform_tests(test_cases)

for (i, result) in enumerate(results)
  test_acc_pruned, num_neurons_pruned, num_neurons_original, same_output = result
  test_case_hidden = test_cases[i][1]
  println("$i: Test case with $test_case_hidden hidden layers")
  if different_output
    println("Different output")
    continue
  end

  println("Test accuracy pruned: ", round(test_acc_pruned, digits=4))
  println("Number of neurons pruned: $(num_neurons_original - num_neurons_pruned) (a $(round((num_neurons_original - num_neurons_pruned) / num_neurons_original * 100, digits=2))% reduction)")
  println("\n")
end
