include("initialisation.jl")

"GLOBAL PARAMETERS"

# random data
nobs, nfeats = 1_000, 5

# Evotrees configuration
forest_size = 500
tree_depth = 5

"TREE TRAINING"

# CONCRETE DATA
evo_model, preds, avg_error = build_forest(tree_depth, forest_size, concrete_data);
x_train, y_train, x_test, y_test = concrete_data();
plot(y_test, [preds, y_test], title="Concrete data", label=["Prediction" "Data"], markershape=[:circle :none], seriestype=[:scatter :line], lw=3)

# L2-NORM (SQUARED) DATA
evo_model, preds, avg_error = build_forest(tree_depth, forest_size, random_data);
x_train, y_train, x_test, y_test = random_data();
plot(y_test, [preds, y_test], title="L2-norm (squared) data", label=["Prediction" "Data"], markershape=[:circle :none], seriestype=[:scatter :line], lw=3)

"OPTIMIZATION"

@time x_new, sol_new, m_new = trees_to_relaxed_MIP(evo_model, true, tree_depth)
@time x_alg, sol_alg, m_algo = trees_to_relaxed_MIP(evo_model, false, tree_depth)

EvoTrees.predict(evo_model, reshape([mean(x_new[n]) for n in 1:nfeats], 1, nfeats))[1]
EvoTrees.predict(evo_model, reshape([mean(x_alg[n]) for n in 1:nfeats], 1, nfeats))[1]
EvoTrees.predict(evo_model, reshape(zeros(nfeats), 1, nfeats))[1]
minimum(preds)

sol_new
sol_alg

"PLOTTING"

trees = [1, 5, 10, 20, 100, 200]
depths = [3, 5, 7, 9]

plot_model_quality(trees, depths, "Tree model NRMSE for concrete data set", "Forest size", "Tree depth", concrete_data)
plot_model_quality(trees, depths, "Tree model NRMSE for L2-norm (squared) data set", "Forest size", "Tree depth", random_data)