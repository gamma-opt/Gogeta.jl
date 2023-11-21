using Flux
using Flux: crossentropy, params, onehotbatch, onecold, throttle, train!, logitcrossentropy
using Flux.Data: DataLoader
using MLDatasets
using Statistics
using Metal
using NPZ

include("./lossless_compression.jl")
include("../bound_tightening.jl")

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

function kaiming_initialization(out, inp)
  scaling = sqrt(2.0 / inp) |> Float64
  return scaling * randn(rng, Float64, (Int64(out), Int64(inp)))
end

function create_model(n_neurons, subdir_path)
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

  num = 0
  for p in params(model)
    num += sum(p .== 0)
  end
  println("Number of zeros: $num")

  return model
end

function get_bounds(subdir_path, model, n_neurons)
  # n_neurons_initial = sum(n_neurons)
  # upper_path = joinpath(subdir_path, "upper_new3.npy")
  # lower_path = joinpath(subdir_path, "lower_new3.npy")
  # upper = npzread(upper_path)
  # lower = npzread(lower_path)
  # U_bounds = Float64[-0.5, 0.5, [1000000 for _ in 3:n_neurons_initial]...]
  # L_bounds = Float64[-1.5, -0.5, [-1000000 for _ in 3:n_neurons_initial]...]

  # best_U_bounds, best_L_bounds = bound_tightening_threads(model, U_bounds, L_bounds, false, gurobi_env)
  # demo_best_U_bounds, demo_best_L_bounds = bound_tightening_threads(model, demo_U_bounds, demo_L_bounds, false, gurobi_env)
  # demo2_best_U_bounds, demo2_best_L_bounds = bound_tightening_threads_old(model, demo_U_bounds, demo_L_bounds, false)
  # demo3_best_U_bounds, demo3_best_L_bounds = bound_tightening(model, demo_U_bounds, demo_L_bounds, false, true, gurobi_env)

  # U1, U2 = U_first_layers[3:1026], U_first_layers[1027:1538]
  # L1, L2 = L_first_layers[3:1026], L_first_layers[1027:1538]

  # upper_new = [U_first_layers[1:2]..., U1..., U2..., U_first_layers[1539:end]...]
  # lower_new = [L_first_layers[1:2]..., L1..., L2..., L_first_layers[1539:end]...]

  # npzwrite(joinpath(subdir_path, "upper_new_new.npy"), upper_new)
  # npzwrite(joinpath(subdir_path, "lower_new_new.npy"), lower_new)
  # npzwrite(joinpath(subdir_path, "upper_new_lp.npy"), demo3_best_U_bounds)
  # npzwrite(joinpath(subdir_path, "lower_new_lp.npy"), demo3_best_L_bounds)

  # # get number of elements where upper is positive and demo is negative
  # for i in 1027:1538
  #   if upper[i] > 0 && demo3_best_U_bounds[i] < 0
  #     println("$(i): $(upper[i]), $(demo_best_U_bounds[i])")
  #   elseif upper[i] < 0 && demo3_best_U_bounds[i] > 0
  #     println("$(i): $(upper[i]), $(demo_best_U_bounds[i])")
  #   end
  # end

  upper = npzread(joinpath(subdir_path, "upper_new_lp.npy"))
  lower = npzread(joinpath(subdir_path, "lower_new_lp.npy"))

  return upper, lower
end

function find_lp_bounds(n_neurons, parent_dir)
  subdirs = filter(d -> isdir(joinpath(parent_dir, d)), readdir(parent_dir))
  n_neurons_initial = sum(n_neurons)

  for subdir in subdirs
    subdir_path = joinpath(parent_dir, subdir)
    println("Test with $subdir_path")

    model = create_model(n_neurons, subdir_path)

    U_bounds = Float64[-0.5, 0.5, [1000000 for _ in 3:n_neurons_initial]...]
    L_bounds = Float64[-1.5, -0.5, [-1000000 for _ in 3:n_neurons_initial]...]

    demo3_best_U_bounds, demo3_best_L_bounds = bound_tightening(model, U_bounds, L_bounds, false, true, gurobi_env)

    npzwrite(joinpath(subdir_path, "upper_new_lp.npy"), demo3_best_U_bounds)
    npzwrite(joinpath(subdir_path, "lower_new_lp.npy"), demo3_best_L_bounds)
  end
end



# constants
n_neurons = Int64[2, 1024, 512, 512, 256, 1]
n_neurons_cumulative_indices = [i+1 for i in [0, cumsum(n_neurons)...]]
parent_dir = "/Users/vimetoivonen/code/school/kandi/train_network/layer_weights"
subdirs = filter(d -> isdir(joinpath(parent_dir, d)), readdir(parent_dir))
i, subdir = 1, subdirs[1]
subdir_path = joinpath(parent_dir, subdir)
println("Test with $subdir_path")

find_lp_bounds(n_neurons, parent_dir)

model = create_model(n_neurons, subdir_path)

upper, lower = get_bounds(subdir_path, model, n_neurons)

# calculate the upper and lower bounds for each layer
upper_bounds = [upper[i:j-1] for (i, j) in zip(n_neurons_cumulative_indices[1:end-1], n_neurons_cumulative_indices[2:end])][2:end]
lower_bounds = [lower[i:j-1] for (i, j) in zip(n_neurons_cumulative_indices[1:end-1], n_neurons_cumulative_indices[2:end])][2:end]
println("The bound lengths: $(map(x -> length(x), upper_bounds))")



# ! ######################################## ! #
# ! # Setup done now we can start pruning  # ! #
# ! ######################################## ! #



model_params = params(model)
weights = [copy(model_params[i]) for i in 1:2:length(model_params)]
biases = [copy(model_params[i]) for i in 2:2:length(model_params)]

# TODO: fix warning
pruned_weights = [copy(weights[i]) for i in 1:length(weights)]
pruned_biases = [copy(biases[i]) for i in 1:length(biases)]


for param_index in 1:length(weights)-1
    W1 = copy(pruned_weights[param_index])
    b1 = copy(pruned_biases[param_index])

    W2 = copy(weights[param_index+1])
    b2 = copy(biases[param_index+1])

    upper1 = copy(upper_bounds[param_index])
    upper2 = copy(upper_bounds[param_index+1])
    lower1 = copy(lower_bounds[param_index])
    lower2 = copy(lower_bounds[param_index+1])

    x = param_index == 1 ? [-1, 0] : rand(rng, Float64, size(W1, 2))
    _, a2_original = forward_pass(x, W1, b1, W2, b2)

    # prune the layer
    W1_pruned, b1_pruned, W2_pruned, b2_pruned = prune_from_model(W1, b1, W2, b2, upper1, upper2, lower1, lower2)

    _, a2_pruned = forward_pass(x, W1_pruned, b1_pruned, W2_pruned, b2_pruned)
    println("The difference between the forward passes")
    println("$(maximum(a2_original - a2_pruned)), $(minimum(a2_original - a2_pruned))")

    pruned_weights[param_index] = copy(W1_pruned)
    pruned_biases[param_index] = copy(b1_pruned)

    pruned_weights[param_index+1] = copy(W2_pruned)
    pruned_biases[param_index+1] = copy(b2_pruned)
end

for (i, (w, b)) in enumerate(zip(pruned_weights, pruned_biases))
    println("$(i): $(size(w)), $(size(b))")
end

# recreate the model
n_neurons_pruned = [2, map(x -> size(x, 1), pruned_weights)...]
pruned_layers = []
for i = 1:length(n_neurons_pruned)-2
    W = pruned_weights[i]
    b = pruned_biases[i]
    layer = Dense(W, b, relu)
    push!(pruned_layers, layer)
end
W_final = pruned_weights[end]
b_final = pruned_biases[end]
push!(pruned_layers, Dense(W_final, b_final))

model_pruned = Chain(pruned_layers...)

# create 10 numbers randomly x1 from [-1.5, -0.5] and x2 from [-0.5, 0.5]
x1 = rand(rng, Float32, 10) .- 1.5
x2 = rand(rng, Float32, 10) .- 0.5
x = hcat(x1, x2)'

# forward pass through the original model
y = model(x)
y_pruned = model_pruned(x)

println("$(y)")
println("$(y_pruned)")




# function forward_pass_deep(x, m)
#   # create a forward pass function, that takes a relu activated model and returns all the intermediate outputs

#   outputs = []

#   for layer in m.layers
#     x = layer(x)
#     push!(outputs, x)
#   end

#   return outputs
# end

# # println(forward_pass_deep(x, model_pruned)[end][1:10])

# # outputs_model_pruned = forward_pass_deep(x, model_pruned)
# # outputs_model = forward_pass_deep(x, model)

# # x_0 = [-1.5; -0.5]
# # ooo = forward_pass_deep(x_0, model)
# # println(ooo[2][1:10])

# # # compare the first layer outputs
# # differences = outputs_model[2] .- outputs_model_pruned[2]
# # println("$(maximum(differences)), $(minimum(differences))")




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
