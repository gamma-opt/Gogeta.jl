include("initialisation.jl")

"TREE TRAINING"

evo_model, preds, avg_error = build_forest(5, 1000, random_data);
plot(y_test, [preds, y_test], markershape=[:circle :none], seriestype=[:scatter :line], lw=3)

"OPTIMIZATION"

@time x_new, sol_new, m_new = trees_to_relaxed_MIP(evo_model, true, tree_depth)
@time x_alg, sol_alg, m_algo = trees_to_relaxed_MIP(evo_model, false, tree_depth)

EvoTrees.predict(evo_model, reshape([mean(x_new[n]) for n in 1:nfeats], 1, nfeats))[1]
EvoTrees.predict(evo_model, reshape([mean(x_alg[n]) for n in 1:nfeats], 1, nfeats))[1]
EvoTrees.predict(evo_model, reshape(zeros(nfeats), 1, nfeats))[1]
minimum(preds)
sum(minimum(evo_model.trees[tree].pred) for tree in eachindex(evo_model.trees))

sol_new
sol_alg

"PLOTTING"

trees = [1, 5, 10, 20, 100, 200]
depths = [3, 5, 7, 9]

plot_model_quality(trees, depths, "Tree model RMSE for concrete data set", "Forest size", "Tree depth", concrete_data)
plot_model_quality(trees, depths, "Tree model RMSE for L2-norm data", "Forest size", "Tree depth", random_data)