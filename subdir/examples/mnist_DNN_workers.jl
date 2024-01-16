using Logging
using Statistics
using ML_as_MO

# location of the train_mnist_DNN! function
include("helper_functions.jl")

@info "Creating and training a standard ReLU DNN using Flux.jl based on the MNIST digit dataset"

mnist_DNN = Chain( # 866 nodes
    Dense(784, 32, relu),
    Dense(32, 24, relu),
    Dense(24, 16, relu),
    Dense(16, 10)
)
train_mnist_DNN!(mnist_DNN)

# input values bounded to [0, 1] (grayscale pixel value)
# other nodes bounded to [-1000, 1000] (arbitrary sufficiently large bounds that wont change the output)
U_bounds = Float32[if i <= 784 1 else 1000 end for i in 1:866]
L_bounds = Float32[if i <= 784 0 else -1000 end for i in 1:866]

@info "Adding as many workers as there are available threads (number van be varied freely) and Importing ML_as_MO in all workers"
using Distributed
n_threads = Threads.nthreads()
addprocs(n_threads)
workers()

@everywhere begin
    import Pkg
    Pkg.activate(".")
    using ML_as_MO
end

@info "Testing create_JuMP_model() with bt=\"workers\" (there are $(n_threads) workers)"
time1 = @elapsed begin
    mdl_bt_workers = @time create_JuMP_model(mnist_DNN, U_bounds, L_bounds, "workers")
end

@info "Testing create_JuMP_model() with bt=\"2 workers\" (there are at most 2 active workers each using half of the available threads)"
time2 = @elapsed begin
    mdl_bt_2workers = @time create_JuMP_model(mnist_DNN, U_bounds, L_bounds, "2 workers")
end

@info "============= Results =============
Creating the JuMP model with multithreaded bound tightening (Workers): $(round(time1; sigdigits = 3))s
Creating the JuMP model with multithreaded bound tightening (2 Workers): $(round(time2; sigdigits = 3))s"
