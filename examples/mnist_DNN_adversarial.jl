using Logging
using Statistics
using ML_as_MO

# location of the train_mnist_DNN! function
include("helper_functions.jl")

@info "Creating and training a standard ReLU DNN using Flux.jl based on the MNIST digit dataset"

mnist_DNN = Chain( # 842 nodes
    Dense(784, 32, relu),
    Dense(32, 16, relu),
    Dense(16, 10)
)
train_mnist_DNN!(mnist_DNN)

# input values bounded to [0, 1] (grayscale pixel value)
# other nodes bounded to [-1000, 1000] (arbitrary sufficiently large bounds that wont change the output)
U_bounds = Float32[if i <= 784 1 else 1000 end for i in 1:842]
L_bounds = Float32[if i <= 784 0 else -1000 end for i in 1:842]

@info "Below we compare the computational time of generating 10 adversarial
      images based on the MNIST training data set. First, the images are generated without
      bound tightening, and afterwards, the same images are generated with tightened bounds."

@info "Generating 10 adversarial images using the initial bounds U_bounds and L_bounds"

adv_time1, adv_imgs1 = @time create_adversarials(mnist_DNN, U_bounds, L_bounds, 10, "L1")

@info "Solving optimal constraint bounds"

bound_solve_time = @elapsed begin
    optimal_U, optimal_L = @time bound_tightening_threads(mnist_DNN, U_bounds, L_bounds)
end

@info "Calculating tighter bounds and generating the same 10 adversarial images using the improved bounds"

adv_time2, adv_imgs2 = @time create_adversarials(mnist_DNN, optimal_U, optimal_L, 10, "L1")

@info "Average adversarial image generation time with old bounds: $(mean(adv_time1))s
      Average adversarial image generation time with tightened bounds: $(mean(adv_time2))s
      Solving the optimal bounds multi-threaded using Threads: $(bound_solve_time)s"

