using Flux
using Flux: crossentropy, params, onehotbatch, onecold, throttle, train!, logitcrossentropy
using Flux.Data: DataLoader
using MLDatasets
using Statistics
using Metal
using NPZ

include("./lossless_compression.jl")
include("../bound_tightening.jl")

const PATIENCE = 3  # Number of epochs to wait before stopping when accuracy does not improve
const CHECK_EVERY = 5  # Check the test accuracy every 5 epochs
# create function to create the gurobi environment
const gurobi_env = Gurobi.Env()
const rng = MersenneTwister(1234)

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

# abstract type AbstractOptimiser end

# mutable struct Adam32 <: AbstractOptimiser
#   eta::Float32
#   beta::Tuple{Float32, Float32}
#   epsilon::Float32
#   state::IdDict{Any,Any}
# end
# Adam32(η::Real = 0.001, β::Tuple = (0.9, 0.999), ϵ::Real = EPS) = Adam32(η, β, ϵ, IdDict())
# Adam32(η::Real, β::Tuple, state::IdDict) = Adam32(η, β, EPS, state)



function kaiming_initialization(out, inp)
  scaling = sqrt(2.0 / inp) |> Float64
  return scaling * randn(rng, Float64, (Int64(out), Int64(inp)))
end

# function train_model(n_neurons_hidden, optimizer_params, regularization_params, downsize_images = true)
#   x_train, Y_train = MNIST(split=:train)[:]
#   x_test, Y_test = MNIST(split=:test)[:]

#   x_train, x_test = Float32.(x_train), Float32.(x_test)

#   if downsize_images
#     x_train = mapslices(downsample_image, x_train; dims=(1,2))
#     x_test = mapslices(downsample_image, x_test; dims=(1,2))
#   end

#   x_train, x_test = Flux.flatten(x_train), Flux.flatten(x_test)
#   Y_train, Y_test = onehotbatch(Y_train, 0:9), onehotbatch(Y_test, 0:9)

#   batchsize = 2048
#   train_loader = DataLoader((x_train, Y_train) |> gpu, batchsize=batchsize, shuffle=true)
#   test_loader = DataLoader((x_test, Y_test) |> gpu, batchsize=batchsize)

#   n_input_layer = downsize_images ? 14^2 : 28^2
#   n_neurons = Int32[n_input_layer, n_neurons_hidden..., 10]
#   layers = []
#   for i = 1:length(n_neurons)-2
#     # Initialize using Kaiming initialization for weights and zeros for biases
#     W_init = kaiming_initialization(n_neurons[i+1], n_neurons[i])
#     b_init = zeros(Float32, n_neurons[i+1])

#     layer = Dense(W_init, b_init, relu)
#     push!(layers, layer)
#   end

#   W_init = kaiming_initialization(n_neurons[end], n_neurons[end-1])
#   b_init = zeros(n_neurons[end])
#   push!(layers, Dense(W_init, b_init))

#   model = Chain(layers...) |> gpu

#   optimizer_index, o1, o2, o3 = optimizer_params
#   # optimizer_index, o1, o2, o3 = Int32(optimizer_index), Float32(o1), Float32(o2), Float32(o3)
#   if optimizer_index == 1
#     o = Momentum(Float32(o1), Float32(o2))
#     optimizer = Flux.setup(o, model)
#   elseif optimizer_index == 2
#     optimizer = ADAM(o1, (o2, o3)) # |> gpu
#   elseif optimizer_index == 3
#     optimizer = Descent(o1) # |> gpu
#   end

#   # o = Descent(0.01)
#   # o = Adam(0.01, (0.9, 0.999))
#   o = Momentum(0.01, 0.9)
#   optimizer = Flux.setup(o, model)

#   # optimizer = Flux.setup(Descent(0.01), model)

#   # print(typeof(optimizer))

#   l1norm(x) = Float32(sum(abs, x))
#   l2norm(x) = Float32(sum(norm, x))

#   # function create_loss_function(regularization_index, r1)
#   #   if regularization_index == 1
#   #     # L1 regularization
#   #     return (x, y, m) -> logitcrossentropy(m(x), y) + Float32(r1) * Float32(sum(l1norm, Flux.params(m)))
#   #   elseif regularization_index == 2
#   #     # L2 regularization
#   #     return (x, y, m) -> logitcrossentropy(m(x), y) # + Float32(r1) * Float32(sum(l2norm, Flux.params(m)))
#   #   else
#   #     # No regularization
#   #     return (x, y, m) -> logitcrossentropy(m(x), y)
#   #   end
#   # end
#   # function create_regularization(regulariztion_index, r1)
#   #   if regulariztion_index == 1
#   #     # L1 regularization
#   #     return (m) -> Float32(r1) * Float32(sum(l1norm, Flux.params(m)))
#   #   elseif regulariztion_index == 2
#   #     # L2 regularization
#   #     return (m) -> Float32(r1) * Float32(sum(l2norm, Flux.params(m)))
#   #   else
#   #     # No regularization
#   #     return (m) -> 0
#   #   end
#   # end

#   r_index, r1 = regularization_params
#   r_index, r1 = Int32(r_index), Float32(r1)
#   # loss_fn = create_loss_function(r_index, r1) |> gpu
#   # regularization = create_regularization(r_index, r1) |> gpu
#   # prms = params(model) |> gpu

#   loss_values = Float32[]

#   previous_test_acc = 0.0
#   non_improving_epochs = 0

#   epoch = 1

#   while true
#       batch_loss_values = zeros(length(train_loader))
#       for (i, (x, y)) in enumerate(train_loader)
#         loss, grads = Flux.withgradient(model) do m
#           # y_hat = m(x)
#           # Flux.logitcrossentropy(y_hat, y) # + regularization(m)
#           # loss_fn(x, y, m)
#           logitcrossentropy(m(x), y) + (i % 5 == 0 ? 0 : r1) * sum(p -> sum(abs, p), params(m))
#         end
#         Flux.update!(optimizer, model, grads[1])
#         batch_loss_values[i] = loss |> cpu
#           # println("loss")
#           # println(typeof(loss))
#           # println("x")
#           # println(typeof(x))
#           # println("y")
#           # println(typeof(y))
#           # println("prms")
#           # println(typeof(prms))
#           # println("optimizer")
#           # println(typeof(optimizer))
#           # println("optimizer params")
#           # # println(typeof(optimizer.eta))
#           # # println(typeof(optimizer.rho))
#           # loss_value, grads = Flux.withgradient(model) do m
#           #   loss(x, y, m)
#           # end
#           # Flux.update!(optimizer, model, grads[1])

#           # train!(loss, prms, [(x, y)], optimizer)
#           # batch_loss_values[i] = loss(x, y) |> cpu

#       end
#       push!(loss_values, mean(batch_loss_values))

#       if epoch % CHECK_EVERY == 0
#           accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
#           test_acc = mean([accuracy(x_batch, y_batch) for (x_batch, y_batch) in test_loader])
#           # test_loss = mean([loss(x_batch, y_batch) for (x_batch, y_batch) in test_loader])

#           println("Epoch: $epoch, Test Accuracy: $test_acc")

#           # Exit the loop if test accuracy does not improve for PATIENCE epochs
#           if test_acc <= previous_test_acc
#               non_improving_epochs += 1
#               if non_improving_epochs >= PATIENCE
#                   break
#               end
#           else
#               non_improving_epochs = 0
#               previous_test_acc = test_acc
#           end
#       end

#       # # Decay the learning rate every 50 epochs
#       # if epoch % 50 == 0
#       #     optimizer.eta *= 0.1
#       # end

#       epoch += 1
#   end

#   model = cpu(model)

#   println("Training finished after $epoch epochs")

#   # round the weights smaller than 1e-5 to 0
#   for p in params(model)
#     p[abs.(p) .< 1e-7] .= 0
#   end

#   return model, test_loader, n_neurons
# end

# function perform_tests(test_cases)
#   results = []

#   for (i, test_case) in enumerate(test_cases)
#     # [hidden_layers], (1: Momentum, 2: ADAM, 3: SGD, lr, momentum), (1: L1, 2: L2, 3: None, regularization strength), downsize_images
#     model_params, optimizer_params, regularization_params, downsize_images = test_case

#     println("$i: Test with $model_params hidden layers")

#     model, test_loader, n_neurons = train_model(model_params, optimizer_params, regularization_params, downsize_images)

#     n_zero_weights = sum(p -> sum(abs.(p) .< 1e-10), params(model))
#     println("Number of zero weights: ($n_zero_weights / $(sum(length, params(model))))")

#     n_neurons_initial = sum(n_neurons)
#     U_bounds = Float32[if i <= n_neurons[1] 1 else 10000 end for i in 1:n_neurons_initial]
#     L_bounds = Float32[if i <= n_neurons[1] 0 else -10000 end for i in 1:n_neurons_initial]

    # best_U_bounds, best_L_bounds = bound_tightening_threads(model, U_bounds, L_bounds, false, gurobi_env)

    # best_U_bounds = best_U_bounds[n_neurons[1] + 1:end]
    # best_L_bounds = best_L_bounds[n_neurons[1] + 1:end]

    # W1, b1, W2, b2, W3, b3 = params(model)

    # upper1 = best_U_bounds[1:n_neurons[2]]
    # upper2 = best_U_bounds[n_neurons[2] + 1:n_neurons[2] + n_neurons[3]]
    # upper3 = best_U_bounds[n_neurons[2] + n_neurons[3] + 1:end]

    # lower1 = best_L_bounds[1:n_neurons[2]]
    # lower2 = best_L_bounds[n_neurons[2] + 1:n_neurons[2] + n_neurons[3]]
    # lower3 = best_L_bounds[n_neurons[2] + n_neurons[3] + 1:end]

    #  # TODO: update these to work with more layers
    # W1_pruned, b1_pruned, W2_pruned, b2_pruned = prune_from_model(W1, b1, W2, b2, upper1, upper2, lower1, lower2)
    # W2_pruned2, b2_pruned2, W3_pruned, b3_pruned = prune_from_model(W2_pruned, b2_pruned, W3, b3, upper2, upper3, lower2, lower3)
#     layer_parameters = [(W1_pruned, b1_pruned), (W2_pruned2, b2_pruned2), (W3_pruned, b3_pruned)]

#     n_neurons_pruned = [size(layer_parameters[1][1], 2), map(x -> size(x[1], 1), layer_parameters)...]

#     layers = []
#     for i = 1:length(n_neurons_pruned)-2
#       layer = Dense(n_neurons_pruned[i], n_neurons_pruned[i+1], relu)
#       push!(layers, layer)
#     end
#     push!(layers, Dense(n_neurons_pruned[end-1], n_neurons_pruned[end]))

#     model_pruned = Chain(layers...)

#     for (i, (w, b)) in enumerate(layer_parameters)
#       model_pruned.layers[i].weight .= w
#       model_pruned.layers[i].bias .= b
#     end

#     num_neurons_original = sum(length, params(model))
#     num_neurons_pruned = sum(length, params(model_pruned))
#     # println("Number of neurons pruned: $(num_neurons_original - num_neurons_pruned) (a $(round((num_neurons_original - num_neurons_pruned) / num_neurons_original * 100, digits=2))% reduction)")

#     accuracy(x, y, m) = mean(onecold(m(x)) .== onecold(y))
#     test_acc_original = mean([accuracy(x_batch, y_batch, model) for (x_batch, y_batch) in test_loader])
#     test_acc_pruned = mean([accuracy(x_batch, y_batch, model_pruned) for (x_batch, y_batch) in test_loader])

#     if (round(test_acc_original, digits=4) != round(test_acc_pruned, digits=4))
#       push!(results, (test_acc_pruned, num_neurons_pruned, num_neurons_original, false))
#     end

#     push!(results, (test_acc_pruned, num_neurons_pruned, num_neurons_original, true))
#   end

#   return results
# end

# test_cases = [
#   [[80, 80], (1, 0.01, 0.9, 0), (1, 0.005), true] #,
#   # [[100, 25], (1, 0.01, 0.9, 0), (1, 0.005), true],
# ]

# results = perform_tests(test_cases)

# for (i, result) in enumerate(results)
#   test_acc_pruned, num_neurons_pruned, num_neurons_original, same_output = result
#   test_case_hidden = test_cases[i][1]
#   println("$i: Test case with $test_case_hidden hidden layers")
#   if !same_output
#     println("Different output")
#     continue
#   end

#   println("Test accuracy pruned: ", round(test_acc_pruned, digits=4))
#   println("Number of parameters pruned: $(num_neurons_original - num_neurons_pruned) (a $(round((num_neurons_original - num_neurons_pruned) / num_neurons_original * 100, digits=2))% reduction)")
#   println("\n")
# end

# This test case returns
# 1: Test case with [100, 25] hidden layers
# Test accuracy pruned: 0.9631
# Number of parameters pruned: 15057 (a 66.96% reduction)


# 2: Test case with [100, 25] hidden layers
# Test accuracy pruned: 0.8808
# Number of parameters pruned: 20522 (a 91.27% reduction)


# n_neurons = Int64[2, 2048,1024,1024,512, 1]
n_neurons = Int64[2, 1024, 512, 512, 256, 1]
n_neurons_cumulative_indices = [i+1 for i in [0, cumsum(n_neurons)...]]

parent_dir = "/Users/vimetoivonen/code/school/kandi/train_network/layer_weights"
subdirs = filter(d -> isdir(joinpath(parent_dir, d)), readdir(parent_dir))

# for (i, subdir) in enumerate(subdirs)
i, subdir = 1, subdirs[1]
subdir_path = joinpath(parent_dir, subdir)

println("$(i): Test with $subdir_path")

upper = npzread(joinpath(parent_dir, subdir, "upper_new3.npy"))
lower = npzread(joinpath(parent_dir, subdir, "lower_new3.npy"))

# if sum(upper .== 10000) != length(upper)
#     continue
# end

# println("$(i): Test with $subdir_path")

weight_paths = [joinpath(subdir_path, "layer_$(i)_weights.npy") for i in 0:4]
bias_paths = [joinpath(subdir_path, "layer_$(i)_biases.npy") for i in 0:4]

layers = []
for i in 1:length(n_neurons)-2
    W = Float64.(npzread(weight_paths[i]))
    b = Float64.(npzread(bias_paths[i]))

    push!(layers, Dense(W, b, relu))
end
W_final = Float64.(npzread(weight_paths[end]))
b_final = Float64.(npzread(bias_paths[end]))
push!(layers, Dense(W_final, b_final))

model = Chain(layers...)

for p in params(model)
  p[abs.(p) .< 1e-7] .= 0
end

# round all weights smaller than 1e-7 to 0
# for p in params(model)
#   p[abs.(p) .< 1e-7] .= 0
# end

n_neurons_initial = sum(n_neurons)

U_bounds = Float64[-0.5, 0.5, [if i <= n_neurons[1] 1 else 1000000 end for i in 3:n_neurons_initial]...]
L_bounds = Float64[-1.5, -0.5, [if i <= n_neurons[1] 0 else -1000000 end for i in 3:n_neurons_initial]...]

println("$(i): Test with $subdir_path")
println("$(length(U_bounds)), $(length(L_bounds))")
println(U_bounds[1:10])
println(L_bounds[1:10])
println(n_neurons_initial)

# best_U_bounds, best_L_bounds = bound_tightening_threads(model, U_bounds, L_bounds, false, gurobi_env)
best_U_bounds, best_L_bounds = bound_tightening(model, U_bounds, L_bounds, false)

npzwrite(joinpath(subdir_path, "upper_new_single.npy"), best_U_bounds)
npzwrite(joinpath(subdir_path, "lower_new_single.npy"), best_L_bounds)

upper = best_U_bounds
lower = best_L_bounds

best_U_bounds_second, best_U_bounds_third = best_U_bounds[3:1026], best_U_bounds[1027:1538]
best_L_bounds_second, best_L_bounds_third = best_L_bounds[3:1026], best_L_bounds[1027:1538]

println("$(length(best_U_bounds_second)), $(length(best_L_bounds_second))")
println("$(length(best_U_bounds_third)), $(length(best_L_bounds_third))")

# compare the second layer upper bounds calculated with upper
upper_second, upper_third = upper[3:1026], upper[1027:1538]
lower_second, lower_third = lower[3:1026], lower[1027:1538]

distances_U_second = best_U_bounds_second .- upper_second
distances_L_second = best_L_bounds_second .- lower_second

distances_U_third = best_U_bounds_third .- upper_third
distances_L_third = best_L_bounds_third .- lower_third

println("$(maximum(distances_U_second)), $(maximum(distances_L_second))")
println("$(maximum(distances_U_third)), $(maximum(distances_L_third))")

# create new bounds contains the new second and third layer bounds
upper_new = [upper[1:2]..., best_U_bounds_second..., best_U_bounds_third..., upper[1539:end]...]
lower_new = [lower[1:2]..., best_L_bounds_second..., best_L_bounds_third..., lower[1539:end]...]

npzwrite(joinpath(subdir_path, "upper_new.npy"), upper_new)
npzwrite(joinpath(subdir_path, "lower_new.npy"), lower_new)

upper_new = npzread(joinpath(subdir_path, "upper_new.npy"))
lower_new = npzread(joinpath(subdir_path, "lower_new.npy"))

# println(maximum(best_L_bounds))
# println(minimum(best_L_bounds))
# println(sum(best_L_bounds .== -10000))
# println(findall(x -> x == 10000, best_U_bounds))

# upper = best_U_bounds
# lower = best_L_bounds

upper_bounds = [upper[i:j-1] for (i, j) in zip(n_neurons_cumulative_indices[1:end-1], n_neurons_cumulative_indices[2:end])][2:end]
lower_bounds = [lower[i:j-1] for (i, j) in zip(n_neurons_cumulative_indices[1:end-1], n_neurons_cumulative_indices[2:end])][2:end]

println(upper_bounds[1][end-10:end])
println(lower_bounds[1][end-10:end])

println(length(upper_bounds))

model_params = params(model)
weights = [model_params[i] for i in 1:2:length(model_params)]
biases = [model_params[i] for i in 2:2:length(model_params)]

pruned_weights = []
pruned_biases = []

# add the first layer weights and biases
push!(pruned_weights, weights[1])
push!(pruned_biases, biases[1])

original_parameters = [(w, b) for (w, b) in zip(weights, biases)]
pruned_parameters = [(w, b) for (w, b) in zip(weights, biases)]

for (i, (w, b)) in enumerate(pruned_parameters)
    println("$(i): $(size(w)), $(size(b))")
end

for param_index in 1:length(pruned_parameters) - 1
    W1 = copy(pruned_weights[param_index])
    b1 = copy(pruned_biases[param_index])

    W2 = copy(weights[param_index+1])
    b2 = copy(biases[param_index+1])

    upper1 = copy(upper_bounds[param_index])
    upper2 = copy(upper_bounds[param_index+1])
    lower1 = copy(lower_bounds[param_index])
    lower2 = copy(lower_bounds[param_index+1])

    if param_index == 1
      x = [-1, 0]
    else
      # create random x of shape size(W1, 2)
      x = rand(rng, Float32, size(W1, 2))
    end

    _, a2_ = forward_pass(x, W1, b1, W2, b2)

    W1_pruned, b1_pruned, W2_pruned, b2_pruned = prune_from_model(W1, b1, W2, b2, upper1, upper2, lower1, lower2)

    _, a2_pruned = forward_pass(x, W1_pruned, b1_pruned, W2_pruned, b2_pruned)

    println("The difference between the forward passes")
    println("$(maximum(a2_ - a2_pruned)), $(minimum(a2_ - a2_pruned))")

    pruned_weights[param_index] = copy(W1_pruned)
    pruned_biases[param_index] = copy(b1_pruned)

    if length(pruned_weights) == param_index
        push!(pruned_weights, copy(W2_pruned))
        push!(pruned_biases, copy(b2_pruned))
    else
        pruned_weights[param_index+1] = copy(W2_pruned)
        pruned_biases[param_index+1] = copy(b2_pruned)
    end
end

for (i, (w, b)) in enumerate(zip(pruned_weights, pruned_biases))
    println("$(i): $(size(w)), $(size(b))")
end

n_neurons_pruned = [2, map(x -> size(x, 1), pruned_weights)...]
println(n_neurons_pruned)

layers = []
for i = 1:length(n_neurons_pruned)-2
    W = pruned_weights[i]
    b = pruned_biases[i]
    layer = Dense(W, b, relu)
    push!(layers, layer)
end
W_final = pruned_weights[end]
b_final = pruned_biases[end]
push!(layers, Dense(W_final, b_final))

model_pruned = Chain(layers...)

# create 10 numbers randomly x1 from [-1.5, -0.5] and x2 from [-0.5, 0.5]
x1 = rand(rng, Float32, 10) .- 1.5
x2 = rand(rng, Float32, 10) .- 0.5

println("min x1: $(minimum(x1)), max x1: $(maximum(x1))")
println("min x2: $(minimum(x2)), max x2: $(maximum(x2))")

x = hcat(x1, x2)'


function forward_pass_deep(x, m)
  # create a forward pass function, that takes a relu activated model and returns all the intermediate outputs

  outputs = []

  for layer in m.layers
    x = layer(x)
    push!(outputs, x)
  end

  return outputs
end

# println(forward_pass_deep(x, model_pruned)[end][1:10])

# outputs_model_pruned = forward_pass_deep(x, model_pruned)
# outputs_model = forward_pass_deep(x, model)

# x_0 = [-1.5; -0.5]
# ooo = forward_pass_deep(x_0, model)
# println(ooo[2][1:10])

# # compare the first layer outputs
# differences = outputs_model[2] .- outputs_model_pruned[2]
# println("$(maximum(differences)), $(minimum(differences))")


# forward pass through the original model
y = model(x)
y_pruned = model_pruned(x)

println("$(y)")
println("$(y_pruned)")

num = 0
for p in params(model)
  num += sum(abs.(p) .< 1e-7)
end
println("Number of zero weights: $num")




    # prune the model using
    # # W1_pruned, b1_pruned, W2_pruned, b2_pruned = prune_from_model(W1, b1, W2, b2, upper1, upper2, lower1, lower2)
    # W2_pruned2, b2_pruned2, W3_pruned, b3_pruned = prune_from_model(W2_pruned, b2_pruned, W3, b3, upper2, upper3, lower2, lower3)



#     # # round all weights smaller than 1e-10 to 0
#     # for p in params(model)
#     #   p[abs.(p) .< 1e-7] .= 0
#     # end

#     # n_neurons_initial = sum(n_neurons)
#     # U_bounds = Float64[if i <= n_neurons[1] 1 else 10000 end for i in 1:n_neurons_initial]
#     # L_bounds = Float64[if i <= n_neurons[1] 0 else -10000 end for i in 1:n_neurons_initial]

#     # best_U_bounds, best_L_bounds = bound_tightening_threads(model, U_bounds, L_bounds, false, gurobi_env)

#     # npzwrite(joinpath(subdir_path, "upper.npy"), best_U_bounds)
#     # npzwrite(joinpath(subdir_path, "lower.npy"), best_L_bounds)
# end




# layers = []

# file_path_prefix = "/Users/vimetoivonen/code/school/kandi/train_network/layer_weights/model_Adam_0_0_0"
# weight_paths = ["layer_0_weights.npy", "layer_1_weights.npy", "layer_2_weights.npy", "layer_3_weights.npy", "layer_4_weights.npy"]
# bias_paths = ["layer_0_biases.npy", "layer_1_biases.npy", "layer_2_biases.npy", "layer_3_biases.npy", "layer_4_biases.npy"]

# for i = 1:length(n_neurons)-2
#   W = Float64.(npzread(joinpath(file_path_prefix, weight_paths[i])))
#   b = Float64.(npzread(joinpath(file_path_prefix, bias_paths[i])))
#   # W = kaiming_initialization(n_neurons[i+1], n_neurons[i])
#   # b = zeros(Float64, n_neurons[i+1])

#   layer = Dense(W, b, relu)
#   push!(layers, layer)
# end

# W_final = Float64.(npzread(joinpath(file_path_prefix, weight_paths[end])))
# b_final = Float64.(npzread(joinpath(file_path_prefix, bias_paths[end])))
# # W_final = kaiming_initialization(n_neurons[end], n_neurons[end-1])
# # b_final = zeros(Float64, n_neurons[end])
# push!(layers, Dense(W_final, b_final))

# model = Chain(layers...)

# n_neurons_initial = sum(n_neurons)
# U_bounds = Float64[if i <= n_neurons[1] 1 else 10000 end for i in 1:n_neurons_initial]
# L_bounds = Float64[if i <= n_neurons[1] 0 else -10000 end for i in 1:n_neurons_initial]

# best_U_bounds, best_L_bounds = bound_tightening_threads(model, U_bounds, L_bounds, false, gurobi_env)

# n_10000_U = sum(best_U_bounds .== 10000)
# n_10000_L = sum(best_L_bounds .== -10000)
