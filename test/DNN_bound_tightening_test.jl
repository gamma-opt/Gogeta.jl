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

@info "Calculating tightened bounds singlethreaded (in-place JuMP model)"
L_singlethread, U_singlethread = bound_tightening(mnist_DNN, U_bounds, L_bounds, false)

@info "Calculating tightened bounds using threads (new JuMP model for each subproblem)"
L_threads, U_threads = bound_tightening_threads(mnist_DNN, U_bounds, L_bounds, false)

@info "Calculating tightened bounds using workers (new JuMP model for each subproblem)
       Note! New workers are not created in this test"
L_workers, U_workers = bound_tightening_workers(mnist_DNN, U_bounds, L_bounds, false)

@info "Calculating tightened bounds using 2 workers (in-place JuMP models for lower and upper bounds respectively)
       Note! New workers are not created in this test"
L_2workers, U_2workers = bound_tightening_2workers(mnist_DNN, U_bounds, L_bounds, false)

@info "Testing that the lower and upper bounds are same for each BT-procedure"
for i in 1:834
    @test L_singlethread[i] ≈ L_threads[i]
    @test L_singlethread[i] ≈ L_workers[i]
    @test L_singlethread[i] ≈ L_2workers[i]

    @test U_singlethread[i] ≈ U_threads[i]
    @test U_singlethread[i] ≈ U_workers[i]
    @test U_singlethread[i] ≈ U_2workers[i]
end

@test true