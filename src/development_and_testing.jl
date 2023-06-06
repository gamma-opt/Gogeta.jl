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
    nn3 = Chain( # 894 nodes
        Dense(784, 40, relu),
        Dense(40, 30, relu),
        Dense(30, 20, relu),
        Dense(20, 10, relu),
        Dense(10, 10)
    )
    return nn1, nn2, nn3
end

raw_nn1, raw_nn2, raw_nn3 = test_nns(42)

nn1, acc1 = train_mnist_nn(raw_nn1)
nn2, acc2 = train_mnist_nn(raw_nn2)
nn3, acc3 = train_mnist_nn(raw_nn3)

bad_U1 = Float32[if i <= 784 1 else 1000 end for i in 1:842]
bad_L1 = Float32[if i <= 784 0 else -1000 end for i in 1:842]
@time optimal_L1, optimal_U1 = solve_optimal_bounds(nn1, bad_U1, bad_L1)

bad_U2 = Float32[if i <= 784 1 else 1000 end for i in 1:866]
bad_L2 = Float32[if i <= 784 0 else -1000 end for i in 1:866]
@time optimal_L2, optimal_U2 = solve_optimal_bounds(nn2, bad_U2, bad_L2)

bad_U3 = Float32[if i <= 784 1 else 1000 end for i in 1:894]
bad_L3 = Float32[if i <= 784 0 else -1000 end for i in 1:894]
@time optimal_L3, optimal_U3 = solve_optimal_bounds(nn3, bad_U3, bad_L3)


bad_times1, bad_imgs1 = create_adversarials(nn1, bad_U1, bad_L1, 1, 10)
optimal_times1, optimal_imgs1 = create_adversarials(nn1, optimal_U1, optimal_L1, 1, 10)
difference1 = bad_times1 - optimal_times1

bad_times2, bad_imgs2 = create_adversarials(nn2, bad_U2, bad_L2, 1, 10)
optimal_times2, optimal_imgs2 = create_adversarials(nn2, optimal_U2, optimal_L2, 1, 10)
difference2 = bad_times2 - optimal_times2

bad_times3, bad_imgs3 = create_adversarials(nn3, bad_U3, bad_L3, 1, 10)
optimal_times3, optimal_imgs3 = create_adversarials(nn3, optimal_U3, optimal_L3, 1, 10)
difference3 = bad_times3 - optimal_times3

@time test = create_JuMP_model(nn1, bad_U1, bad_L1, true)

test_time, test_img = create_adversarials(nn1, bad_U1, bad_L1, 1, 1, "L1", true)