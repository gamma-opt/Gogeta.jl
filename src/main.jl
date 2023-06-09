include("initialisation.jl")

tree_depth = 5
forest_size = 500
Random.seed!(2)

nobs, nfeats = 1_000, 5
x_train = randn(nobs, nfeats)
y_train = Array{Float64}(undef, nobs)
[y_train[i] = sum(x_train[i,:].^2) for i = 1:nobs]

x_test = randn(nobs, nfeats)
y_test = Array{Float64}(undef, nobs)
[y_test[i] = sum(x_test[i,:].^2) for i = 1:nobs]

config = EvoTreeRegressor(max_depth=tree_depth, nbins=32, nrounds=forest_size, loss=:linear)
evo_model = fit_evotree(config; x_train, y_train)
preds = EvoTrees.predict(evo_model, x_test)
avg_error = rms(preds, y_test) / mean(y_test)

plot(y_test, [preds, y_test], markershape=[:circle :none], seriestype=[:scatter :line], lw=3)

@time sol_new = trees_to_relaxed_MIP(evo_model, tree_depth, tree_depth)
@time sol_algo = trees_to_relaxed_MIP(evo_model, 0, tree_depth)
@time sol_old = GBtrees_MIP(evo_model)

EvoTrees.predict(evo_model, reshape(sol_new.-1e-15, 1, 5))
EvoTrees.predict(evo_model, reshape(sol_algo.-1e-15, 1, 5))
EvoTrees.predict(evo_model, reshape(sol_old.-1e-15, 1, 5))
EvoTrees.predict(evo_model, reshape([0 0 0 0 0], 1, 5))