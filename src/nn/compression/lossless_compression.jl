using LinearAlgebra
using Flux: Chain, Dense, relu
using Random
using NPZ

Random.seed!(1234)

# helper functions
function forward_pass(X, W1_, b1_, W2_, b2_)
  A1 = W1_ * X .+ b1_[:]
  R1 = relu.(A1)
  product = W2_ * R1
  A2 = product .+ b2_[:]
  return A1, A2
end

function rank_threshold(M, threshold=1e-5)
  # TODO: compare these two methods
  return rank(M)

  # F = svd(M)
  # S = F.S
  # return sum(abs.(S) .> threshold)
end

# prune neurons that are always inactive
function prune_by_upper_bound(G1, G_bar1, W1, b1, G2, G_bar2, W2, b2)
  # TODO: consider using leq and a threshold instead of 0 (epsilon)
  to_prune = [i for i = 1:size(W1, 1) if G1[i] < 0]

  W1 = W1[setdiff(1:end, to_prune), :]
  b1 = b1[setdiff(1:end, to_prune)]
  G1 = G1[setdiff(1:end, to_prune)]
  G_bar1 = G_bar1[setdiff(1:end, to_prune)]

  W2 = W2[:, setdiff(1:end, to_prune)]

  return G1, G_bar1, W1, b1, G2, G_bar2, W2, b2, length(to_prune)
end

# prune neurons that have zero weights
function prune_zero_weights(G1, G_bar1, W1, b1, G2, G_bar2, W2, b2)
  # TODO: consider using a threshold instead of 0 (epsilon)
  to_prune = [index[1] for index in findall(sum(abs.(W1), dims=2) .< 1e-5)]

  b2 .+= W2[:, to_prune] * b1[to_prune]

  W1 = W1[setdiff(1:end, to_prune), :]
  b1 = b1[setdiff(1:end, to_prune)]
  G1 = G1[setdiff(1:end, to_prune)]
  G_bar1 = G_bar1[setdiff(1:end, to_prune)]

  W2 = W2[:, setdiff(1:end, to_prune)]

  return G1, G_bar1, W1, b1, G2, G_bar2, W2, b2, length(to_prune)
end

# prune layers that are stable
function prune_stable_layer(W1, b1, W2, b2, S, X)
  n_neurons_l_minus_1 = size(X, 1)
  n_neurons_l_plus_1 = size(W2, 1)

  W_bar = zeros(Float32, n_neurons_l_minus_1, n_neurons_l_plus_1)
  b_bar = zeros(Float32, n_neurons_l_plus_1)

  for i in 1:n_neurons_l_plus_1
      b_bar[i] = b2[i] + dot(W2[i, collect(S)], b1[collect(S)])

      for j in 1:n_neurons_l_minus_1
          W_bar[j, i] = dot(W1[collect(S), j], W2[i, collect(S)])
      end
  end

  return W_bar', b_bar
end

# prune neurons that are linearly dependent on other neurons
function prune_stabily_active(G1, G_bar1, W1, b1, G2, G_bar2, W2, b2, X, S, i)
  W_current = W1[[i; collect(S)], :]

  if rank_threshold(W_current) > length(S)
      push!(S, i)
      return G1, G_bar1, W1, b1, G2, G_bar2, W2, b2, S
  end

  alpha = W1[collect(S), :]' \ W1[i, :]

  for j in axes(W2, 1)
      W2[j, collect(S)] .+= alpha .* W2[j, i]
      b2[j] += W2[j, i] * (b1[i] - dot(alpha, b1[collect(S)]))
  end

  W1 = W1[setdiff(1:end, i), :]
  b1 = b1[setdiff(1:end, i)]
  # G1 = G1[setdiff(1:end, i)]
  # G_bar1 = G_bar1[setdiff(1:end, i)]
  W2 = W2[:, setdiff(1:end, i)]
  S = Set{Int}([idx >= i ? idx - 1 : idx for idx in S])

  return G1, G_bar1, W1, b1, G2, G_bar2, W2, b2, S
end

# prune the network
function prune_neuron(W1, b1, W2, b2, X, G1, G2, G_bar1, G_bar2)
  A1, A2 = forward_pass(X, W1, b1, W2, b2)

  S = Set{Int}()
  pruned = false
  # TODO: start updating the unstable flag when we have a stable layer
  unstable = true

  # TODO: check that it works even if all neurons are pruned
  G1, G_bar1, W1, b1, G2, G_bar2, W2, b2, n_pruned_by_upper_bound = prune_by_upper_bound(G1, G_bar1, W1, b1, G2, G_bar2, W2, b2)
  G1, G_bar1, W1, b1, G2, G_bar2, W2, b2, n_pruned_by_zero_weight = prune_zero_weights(G1, G_bar1, W1, b1, G2, G_bar2, W2, b2)

  n_neurons_initial = size(W1, 1)

  for i in n_neurons_initial:-1:1
      if G_bar1[i] > 1e-5
          s_before = length(S)

          G1, G_bar1, W1, b1, G2, G_bar2, W2, b2, S = prune_stabily_active(G1, G_bar1, W1, b1, G2, G_bar2, W2, b2, X, S, i)

          length(S) == s_before && (pruned = true)
      else
          unstable = true
      end
  end

  n_neurons_final = size(W1, 1)
  n_pruned_by_linear_dependence = n_neurons_initial - n_neurons_final

  _, A2_pruned = forward_pass(X, W1, b1, W2, b2) # TODO: remove this as it is inaccurate in odd bounded layers
  is_close = all(abs.(A2 - A2_pruned) .< 1e-3)

  if !unstable
    if length(S) == 0
      # TODO: update this to return the full pruned network
      Upsilon = A2_pruned
      W2 = zeros(size(W2))
      b2 = fill(Upsilon, size(b2))
      println("Constant output, pruned full layer")
      return W1, b1, W2, b2, pruned, is_close, true
    end

    W_bar, b_bar = prune_stable_layer(W1, b1, W2, b2, S, X)

    # TODO: update this ridiculous naming
    A2_superpruned = W_bar * X .+ b_bar[:]

    return W1, b1, W2, b2, pruned, is_close, is_close && all(abs.(A2 - A2_superpruned) .< 1e-3), n_pruned_by_upper_bound, n_pruned_by_zero_weight, n_pruned_by_linear_dependence
  end

  return W1, b1, W2, b2, pruned, is_close, false, n_pruned_by_upper_bound, n_pruned_by_zero_weight, n_pruned_by_linear_dependence
end

# main function that takes the parameters of consecutive layers and their bounds, and compresses them
function prune_from_model(W1, b1, W2, b2, upper1, upper2, lower1, lower2)
  n_neurons = size(W1, 2)

  X = rand(n_neurons)

  G1, G2 = upper1, upper2
  G_bar1, G_bar2 = lower1, lower2

  W1, b1, W2, b2, pruned, is_close, layer_folded, n_pruned_by_upper_bound, n_pruned_by_zero_weight, n_pruned_by_linear_dependence = prune_neuron(W1, b1, W2, b2, X, G1, G2, G_bar1, G_bar2)

  return W1, b1, W2, b2, n_pruned_by_upper_bound, n_pruned_by_zero_weight, n_pruned_by_linear_dependence
end

# ! FOR DEBUGGING
function add_linear_dependencies_to_matrix(W1, max_dependencies=2)
  n_rows, _ = size(W1)
  num_dependencies = rand(1:max_dependencies)

  for _ in 1:num_dependencies
    dependent_row = rand(1:n_rows)
    num_contributors = rand(1:n_rows-1)

    # get the indices of the rows (all rows except the dependent row), and shuffle them
    contributor_rows = shuffle(setdiff(1:n_rows, dependent_row))[1:num_contributors]

    W1[dependent_row, :] .= 0.0

    for i in 1:num_contributors
      W1[dependent_row, :] .+= rand() * 10 * W1[contributor_rows[i], :]
    end
  end

  return W1
end

function create_matrices(n_input = 3, n_neurons_1 = 3, n_neurons_2 = 3, fraction_of_linear_dependencies = 0.5)
  W1 = ((rand(n_input, n_neurons_1) .- 0.1) .* 2)'
  b1 = (rand(n_neurons_1) .- 0.1) .* 2
  W2 = ((rand(n_neurons_1, n_neurons_2) .- 0.1) .* 2)'
  b2 = (rand(n_neurons_2) .- 0.1) .* 2

  if rand() < fraction_of_linear_dependencies
    W1 = add_linear_dependencies_to_matrix(W1)
  end

  X = (rand(n_input) .- 0.5) .* 2

  return W1, b1, W2, b2, X
end

function create_simple_matrices()
  W1 = Float32[
    1  1  0;
    2  0  -1;
    4  2  -1;
  ]
  b1 = Float32[5, 4, 6]
  W2 = Float32[
    2  1  0;
    0  1  2;
    1  0  2;
  ]
  b2 = Float32[0.2, 0.3, 0.1]

  X = Float32[1.0, 1.0, 1.0]

  return W1, b1, W2, b2, X
end

function run_tests(n_iterations = 10)
  pruned_count = 0
  layers_folded = 0
  different_output_count = 0

  for iter in 1:n_iterations
      W1, b1, W2, b2, X = create_matrices(5, 5, 5, 0.5)
      # W1, b1, W2, b2, X = create_simple_matrices() # use this for manual debugging

      A1, A2 = forward_pass(X, W1, b1, W2, b2)
      # for single outputs this works (G == G_bar)
      G1, G2 = A1, A2
      G_bar1, G_bar2 = A1, A2

      W1, b1, W2, b2, pruned, is_close, layer_folded, _, _, _ = prune_neuron(W1, b1, W2, b2, X, G1, G2, G_bar1, G_bar2)

      pruned && (pruned_count += 1)
      layer_folded && (layers_folded += 1)
      !is_close && (different_output_count += 1)
  end

  println("Number of pruned iterations: $pruned_count")
  println("Number of layers folded: $layers_folded")
  println("Number of iterations where A_2 was different than A_2_pruned: $different_output_count")
end
