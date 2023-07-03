using ML_as_MO
using Random
# This file contains random code snppets that are ONLY used for development and testing

x_train, y_train = MNIST(split=:train)[:]
x_test, y_test = MNIST(split=:test)[:]
function test_nns(seed)
    Random.seed!(seed)
    nn1 = Chain( # 842 nodes
        Dense(784, 32, relu),
        Dense(32, 16, relu),
        Dense(16, 10)
    )
    nn2 = Chain( # 866 nodes
        Dense(784, 32, relu),
        Dense(32, 24, relu),
        Dense(24, 16, relu),
        Dense(16, 10)
    )
    nn3 = Chain( # 884 nodes
        Dense(784, 30, relu),
        Dense(30, 30, relu),
        Dense(30, 20, relu),
        Dense(20, 10, relu),
        Dense(10, 10)
    )
    return nn1, nn2, nn3
end

nn1, nn2, nn3 = test_nns(42)

train_mnist_DNN!(nn1)
train_mnist_DNN!(nn2)
train_mnist_DNN!(nn3)


using Distributed
using SharedArrays

n_threads = Threads.nthreads()
n_cores = Sys.CPU_THREADS
addprocs(n_threads)
nprocs()
workers()
worker = workers()


@everywhere begin
    import Pkg
    Pkg.activate(".")
    # Pkg.instantiate()
    # include("initialisation.jl")
end

@everywhere include("../initialisation.jl")


bad_U1 = Float32[if i <= 784 1 else 1000 end for i in 1:842]
bad_L1 = Float32[if i <= 784 0 else -1000 end for i in 1:842]
bad_U2 = Float32[if i <= 784 1 else 1000 end for i in 1:866]
bad_L2 = Float32[if i <= 784 0 else -1000 end for i in 1:866]
bad_U3 = Float32[if i <= 784 1 else 1000 end for i in 1:884]
bad_L3 = Float32[if i <= 784 0 else -1000 end for i in 1:884]


@time optimal_U1, optimal_L1 = bound_tightening(nn1, bad_U1, bad_L1)
@time optimal_U1_multi, optimal_L1_multi = solve_optimal_bounds_multi(nn1, bad_U1, bad_L1)
@time optimal_U1_threads, optimal_L1_threads = bound_tightening_threads(nn1, bad_U1, bad_L1)

@time pmap_U1, pmap_L1 = bound_tightening_workers(nn1, bad_U1, bad_L1)
@time workers2_U1, workers2_L1 = solve_optimal_bounds_2workers(nn1, bad_U1, bad_L1)





@time optimal_U2, optimal_L2 = bound_tightening(nn2, bad_U2, bad_L2)
@time optimal_U2_multi, optimal_L2_multi = solve_optimal_bounds_multi(nn2, bad_U2, bad_L2)
@time optimal_U2_threads, optimal_L2_threads = bound_tightening_threads(nn2, bad_U2, bad_L2)

@time pmap_U2, pmap_L2 = bound_tightening_workers(nn2, bad_U2, bad_L2)
@time workers2_U2, workers2_L2 = solve_optimal_bounds_2workers(nn2, bad_U2, bad_L2)




@time optimal_U3, optimal_L3 = bound_tightening(nn3, bad_U3, bad_L3)
@time optimal_U3_multi, optimal_L3_multi = solve_optimal_bounds_multi(nn3, bad_U3, bad_L3)
@time optimal_U3_threads, optimal_L3_threads = bound_tightening_threads(nn3, bad_U3, bad_L3)

@time pmap_U3, pmap_L3 = bound_tightening_workers(nn3, bad_U3, bad_L3)
@time workers2_U3, workers2_L3 = solve_optimal_bounds_2workers(nn3, bad_U3, bad_L3)









bad_times1, bad_imgs1 = create_adversarials(nn1, bad_U1, bad_L1, 1, 10)
optimal_times1, optimal_imgs1 = create_adversarials(nn1, optimal_U1, optimal_L1, 1, 10)
difference1 = bad_times1 - optimal_times1

bad_times2, bad_imgs2 = create_adversarials(nn2, bad_U2, bad_L2, 1, 10)
optimal_times2, optimal_imgs2 = create_adversarials(nn2, optimal_U2, optimal_L2, 1, 10)
difference2 = bad_times2 - optimal_times2

bad_times3, bad_imgs3 = create_adversarials(nn3, bad_U3, bad_L3, 1, 1)
optimal_times3, optimal_imgs3 = create_adversarials(nn3, optimal_U3, optimal_L3, 1, 1)
difference3 = bad_times3 - optimal_times3

@time test = create_JuMP_model(nn1, bad_U1, bad_L1, "threads")

test_time, test_img = create_adversarials(nn1, bad_U1, bad_L1, 1, 1, "L1", true)
