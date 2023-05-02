include("initialisation.jl")


config = EvoTreeRegressor(max_depth=5, nbins=32, nrounds=10)
nobs, nfeats = 1_000, 5
x_train, y_train = randn(nobs, nfeats), rand(nobs)
model = fit_evotree(config; x_train, y_train)
preds = EvoTrees.predict(model, x_train)
plot(model, 2)


evo_tree = model.trees[2]
exract_tree_nodes_info(evo_tree)





