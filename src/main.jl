include("initialisation.jl")

tree_depth = 5
forest_size = 100
Random.seed!(1)

nobs, nfeats = 1_000, 5
x_train = randn(nobs, nfeats)
y_train = Array{Float64}(undef, nobs)
[y_train[i] = sum(x_train[i,:].^2) for i = 1:nobs]

x_test = randn(nobs, nfeats)
y_test = Array{Float64}(undef, nobs)
[y_test[i] = sum(x_test[i,:].^2) for i = 1:nobs]

config = EvoTreeRegressor(max_depth=tree_depth, nbins=32, nrounds=forest_size, loss=:linear, T=Float64)
evo_model = fit_evotree(config; x_train, y_train)
preds = EvoTrees.predict(evo_model, x_test)
avg_error = rms(preds, y_test) / mean(y_test)

plot(y_test, [preds, y_test], markershape=[:circle :none], seriestype=[:scatter :line], lw=3)

@time x_new, sol_new = trees_to_relaxed_MIP(evo_model, true, tree_depth)
@time x_alg, sol_alg = trees_to_relaxed_MIP(evo_model, false, tree_depth)
@time x_old, sol_old = GBtrees_MIP(evo_model)

EvoTrees.predict(evo_model, reshape(x_new, 1, 5))[1]
EvoTrees.predict(evo_model, reshape(x_alg, 1, 5))[1]
EvoTrees.predict(evo_model, reshape(x_old, 1, 5))[1]
EvoTrees.predict(evo_model, reshape([0 0 0 0 0], 1, 5))[1]

sol_new
sol_alg
sol_old