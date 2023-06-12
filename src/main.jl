include("initialisation.jl")

tree_depth = 5
forest_size = 1000
Random.seed!(3)

nobs, nfeats = 1_000, 3
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

@time x_new, sol_new, m_new = trees_to_relaxed_MIP(evo_model, true, tree_depth)
@time x_alg, sol_alg, m_algo = trees_to_relaxed_MIP(evo_model, false, tree_depth)

EvoTrees.predict(evo_model, reshape([mean(x_new[n]) for n in 1:nfeats], 1, nfeats))[1]
EvoTrees.predict(evo_model, reshape([mean(x_alg[n]) for n in 1:nfeats], 1, nfeats))[1]
EvoTrees.predict(evo_model, reshape(zeros(nfeats), 1, nfeats))[1]
minimum(preds)
sum(minimum(evo_model.trees[tree].pred) for tree in eachindex(evo_model.trees))

sol_new
sol_alg

# FLOATING POINT ERRORS ?
# INCORRECT DATA EXTRACTION FROM EVOTREES ?
