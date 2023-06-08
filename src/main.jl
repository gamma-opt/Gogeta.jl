include("initialisation.jl")

config = EvoTreeRegressor(max_depth=5, nbins=32, nrounds=100)
nobs, nfeats = 1_000, 5
x_train = randn(nobs, nfeats)
y_train = Array{Float64}(undef, nobs)
[y_train[i] = sum(x_train[i,:].^2) for i = 1:nobs]

evo_model = fit_evotree(config; x_train, y_train)
preds = EvoTrees.predict(evo_model, x_train)
plot(evo_model, 3)

new_model = trees_to_relaxed_MIP(evo_model, 5, 5);

lazy_model = trees_to_relaxed_MIP(evo_model, 0, 5);

old_model = GBtrees_MIP(evo_model);


function print_solution(n_feats, model, n_splits, splitpoints)
    println("\n=========================SOLUTION=========================")
    smallest_splitpoint = Array{Int64}(undef, n_feats)
    for ele in eachindex(model[:x])
        println("FEATURE AND SPLIT: $ele, VALUE: $(value(model[:x][ele]))")
    end
    println("==========================================================\n")
end
