using Test, Logging
using ML_as_MO

# location of the train_mnist_DNN! function
include("../examples/helper_functions.jl")

@info "Creating and training a standard ReLU DNN using Flux.jl based on the MNIST digit dataset"

mnist_DNN = Chain( # 834 nodes
    Dense(784, 24, relu),
    Dense(24, 16, relu),
    Dense(16, 10)
)
train_mnist_DNN!(mnist_DNN)

# input values bounded to [0, 1] (grayscale pixel value)
# other nodes bounded to [-1000, 1000] (arbitrary sufficiently large bounds that wont change the output)
U_bounds = Float32[if i <= 784 1 else 1000 end for i in 1:834]
L_bounds = Float32[if i <= 784 0 else -1000 end for i in 1:834]

# running the function once to compile it (only takes a few seconds)
create_JuMP_model(mnist_DNN, L_bounds, U_bounds, "none")

@info "Testing create_JuMP_model() with bt=\"none\""
time1 = @elapsed begin
    mdl_bt_none = @time create_JuMP_model(mnist_DNN, L_bounds, U_bounds, "none")
end

@info "Testing create_JuMP_model() with bt=\"singlethread\""
time2 = @elapsed begin
    mdl_bt_singlethread = @time create_JuMP_model(mnist_DNN, L_bounds, U_bounds, "singlethread")
end

@info "Testing create_JuMP_model() with bt=\"threads\""
time3 = @elapsed begin
    mdl_bt_threads = @time create_JuMP_model(mnist_DNN, L_bounds, U_bounds, "threads")
end

@info "Testing create_JuMP_model() with bt=\"workers\""
time4 = @elapsed begin
    mdl_bt_workers = @time create_JuMP_model(mnist_DNN, L_bounds, U_bounds, "workers")
end

@info "Testing create_JuMP_model() with bt=\"2 workers\""
time5 = @elapsed begin
    mdl_bt_2workers = @time create_JuMP_model(mnist_DNN, L_bounds, U_bounds, "2 workers")
end

@info "============= Results =============
Creating the JuMP model without bound tightening (default bounds): $(round(time1; sigdigits = 3))s
Creating the JuMP model with single-threaded bound tightening: $(round(time2; sigdigits = 3))s
Creating the JuMP model with multithreaded bound tightening (Threads): $(round(time3; sigdigits = 3))s
Creating the JuMP model with multithreaded bound tightening (Workers): $(round(time4; sigdigits = 3))s
Creating the JuMP model with multithreaded bound tightening (2 Workers): $(round(time5; sigdigits = 3))s"

@test true

