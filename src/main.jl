include("initialisation.jl")

config = EvoTreeRegressor(max_depth=12, nbins=100, nrounds=10)
nobs, nfeats = 1_000, 5
x_train = randn(nobs, nfeats)
y_train = Array{Float64}(undef, nobs)
[y_train[i] = sum(x_train[i,:].^2) for i = 1:nobs]

evo_model = fit_evotree(config; x_train, y_train)
preds = EvoTrees.predict(evo_model, x_train)
plot(evo_model, 3)

@time new_model = trees_to_relaxed_MIP(evo_model, 12, 12);

@time lazy_model = trees_to_relaxed_MIP(evo_model, 0, 12);

@time old_model = GBtrees_MIP(evo_model);


function print_solution(n_feats, model, n_splits, splitpoints)
    println("\n=========================SOLUTION=========================")
    for f = 1:n_feats 
        x_opt = Array{Float64}(undef,  n_splits[f])
        [x_opt[i] = value.(model[:x])[f,i] for i = 1:n_splits[f]]
        first_index = findfirst(x -> x==1, x_opt)
        if first_index === nothing
            println("x_$f is unbound")
        else
            println("x_$f <= $(splitpoints[f][3,first_index])")
        end
    end
    println("==========================================================\n")
end
