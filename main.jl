include("src/initialisation.jl")

# read iris data 

x_train, y_train = MNIST(split=:train)[:]
x_test, y_test = MNIST(split=:test)[:]
x_train_flatten = flatten(x_train)
x_test_flatten = flatten(x_test)

config = EvoTreeRegressor(
    loss=:linear, 
    nrounds=100, 
    max_depth=6, 
    nbins=32,
    eta=0.1,
    lambda=0.1, 
    gamma=0.1, 
    min_weight=1.0,
    rowsample=0.5, 
    colsample=0.8)

m = fit_evotree(config; x_train_flatten, y_train)

preds = m(x_train)