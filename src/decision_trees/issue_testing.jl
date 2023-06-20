using EvoTrees
using Plots
using Random
using Statistics
using Interpolations
include("plotting.jl");

Random.seed!(1)
n_feats = 5
data = randn(1000, n_feats)
data = data[shuffle(1:end), :]

split::Int = floor(0.75 * length(data[:, 1]))

x_train = data[1:split, :];
y_train = Array{Float64}(undef, length(x_train[:, 1]));
[y_train[i] = sqrt(sum(x_train[i, :].^2)) for i in 1:length(y_train)];

x_test = data[split+1:end, :];
y_test = Array{Float64}(undef, length(x_test[:, 1]));
[y_test[i] = sqrt(sum(x_test[i, :].^2)) for i in 1:length(y_test)];

config = EvoTreeRegressor(nrounds=1000, max_depth=5, T=Float64, loss=:linear);
model = fit_evotree(config; x_train, y_train);

pred_train = EvoTrees.predict(model, x_train)
pred_test = EvoTrees.predict(model, x_test)

using JuMP
using Gurobi

include("trees_to_relaxed_op.jl");

x_new, sol_new, m_new = trees_to_relaxed_MIP(model, :createinitial, 5, :min);

x_alg, sol_alg, m_alg = trees_to_relaxed_MIP(model, :noconstraints, 5, :min);


"MANUAL CHECKING OF SOLUTION"

y_solution = Array{Int64}(undef, 1000)

for leaf in eachindex(m_new[:y])
    if value(m_new[:y][leaf]) == 1
        y_solution[leaf[1]] = leaf[2]
    end
end

n_trees = length(model.trees) - 1 # number of trees in the model (excluding the bias tree)
n_feats = length(model.info[:fnames]) # number of features (variables) in the model

n_leaves = Array{Int64}(undef, n_trees) # array for the number of leaves on each tree
leaves = Array{Array}(undef, n_trees) # array for the ids of the leaves for each tree

# Get number of leaves and ids of the leaves on each tree
for tree in 1:n_trees
    leaves[tree] = findall(x -> x != 0, vec(model.trees[tree + 1].pred))
    n_leaves[tree] = length(leaves[tree])
end

prediction = model.trees[1].pred[1]

for tree in 1:n_trees
    for leaf in eachindex(leaves[tree])
        if y_solution[tree] == leaf
            prediction += model.trees[tree + 1].pred[leaves[tree][leaf]]
        end
    end
end

prediction
sol_new

EvoTrees.predict(model, reshape([mean(x_new[n]) for n in 1:n_feats], 1, n_feats))[1]
ans

for tree in 1:n_trees
    ans = reshape([mean(x_new[n]) for n in 1:n_feats], 1, n_feats)
    if EvoTrees.predict(model.trees[tree + 1], ans)[1] != model.trees[tree + 1].pred[leaves[tree][y_solution[tree]]]
        println("TREE: $tree")
    end
end