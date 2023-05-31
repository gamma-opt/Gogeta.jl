include("initialisation.jl")

config = EvoTreeRegressor(max_depth=5, nbins=32, nrounds=10)
nobs, nfeats = 1_000, 5
x_train = randn(nobs, nfeats)
y_train = Array{Float64}(undef, nobs)
[y_train[i] = sum(x_train[i,:].^2) for i = 1:nobs]

evo_model = fit_evotree(config; x_train, y_train)
preds = EvoTrees.predict(evo_model, x_train)
plot(evo_model, 2)

gbmodel = GBtrees_MIP(evo_model)
optimize!(gbmodel)

for f = 1:nfeats 
    x_opt = Array{Float64}(undef,  n_splits[f])
    [ x_opt[i] = value.(model[:x])[f,i] for i = 1:n_splits[f]]
    first_index = findfirst(x -> x==1, x_opt)
    print("x_$f <= $(splits[f][3,first_index]) \n")
end

