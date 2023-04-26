using Flux
using Flux: params, train!, mse
using JuMP
using HiGHS

# packages for MNIST digits
using MLDatasets, CUDA, FileIO
using Flux: train!, onehotbatch, flatten

include("neural_nets.jl")
include("JuMP_model.jl")

function test1()
    sum_model = create_sum_nn(
        100, 5, 2, Chain(
                   Dense(2, 4, relu),
                   Dense(4, 3, relu),
                   Dense(3, 1)), 
    )
    jm = create_JuMP_model(sum_model, "predict")
    inputs = [[0.5;0.5], [0.1;0.9], [0.75;0.75]]
    len = length(inputs)
    for i in 1:len
        evaluate(jm, inputs[i])
        optimize!(jm)
        @assert has_values(jm) "JuMP model infeasible with input $(inputs[i])."

        nn_result = sum_model(inputs[i])[1]
        JuMP_model_result = objective_value(jm)
        println("nn_result: ", nn_result)
        println("JuMP_model_result: ", JuMP_model_result)

        @assert abs(nn_result-JuMP_model_result) < 1e-3 "nn and JuMP model give different results."
    end

    println("Test1 passed")
end

function test2()
    sum_model = create_sum_nn(
        100, 5, 20, Chain(
                    Dense(20, 10, relu),
                    Dense(10, 6, relu),
                    Dense(6, 4, relu),
                    Dense(4, 1)), 
    )
    jm = create_JuMP_model(sum_model, "predict")
    inputs = [[1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1],
              [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0]]
    len = length(inputs)
    for i in 1:len
        evaluate(jm, inputs[i])
        optimize!(jm)
        @assert has_values(jm) "JuMP model infeasible with input $(inputs[i])."

        nn_result = sum_model(inputs[i])[1]
        JuMP_model_result = objective_value(jm)
        println("nn_result: ", nn_result)
        println("JuMP_model_result: ", JuMP_model_result)

        @assert abs(nn_result-JuMP_model_result) < 1e-3 "nn and JuMP model give different results."
    end

    println("Test2 passed")
end

function test3()
    rosenbrock_model = create_rosenbrock_nn(
        100, 5, Chain(
                Dense(2, 50, relu),
                Dense(50, 50, relu),
                Dense(50, 50, relu),
                Dense(50, 50, relu),
                Dense(50, 1)
	    )
    )
    jm = create_JuMP_model(rosenbrock_model, "predict")

    inputs = [[0;0], [1;1], [0.5;0.5]]
    len = length(inputs)
    for i in 1:len
        evaluate(jm, inputs[i])
        optimize!(jm)
        @assert has_values(jm) "JuMP model infeasible with input $(inputs[i])."

        nn_result = rosenbrock_model(inputs[i])[1]
        JuMP_model_result = objective_value(jm)
        println("nn_result: ", nn_result)
        println("JuMP_model_result: ", JuMP_model_result)

        @assert abs(nn_result-JuMP_model_result) < 1e-3 "nn and JuMP model give different results."
    end

    println("Test3 passed")
end

function test4()
    rosenbrock_model = create_rosenbrock_nn(
        100, 5, Chain(
                Dense(2, 10, relu),
                Dense(10, 10, relu), Dense(10, 10, relu),
                Dense(10, 10, relu), Dense(10, 10, relu),
                Dense(10, 10, relu), Dense(10, 10, relu),
                Dense(10, 1)
	    )
    )
    jm = create_JuMP_model(rosenbrock_model, "predict")

    inputs = [[0;0], [1;1], [0.5;0.5]]
    len = length(inputs)
    for i in 1:len
        evaluate(jm, inputs[i])
        optimize!(jm)
        @assert has_values(jm) "JuMP model infeasible with input $(inputs[i])."

        nn_result = rosenbrock_model(inputs[i])[1]
        JuMP_model_result = objective_value(jm)
        println("nn_result: ", nn_result)
        println("JuMP_model_result: ", JuMP_model_result)

        @assert abs(nn_result-JuMP_model_result) < 1e-3 "nn and JuMP model give different results."
    end

    println("Test4 passed")
end

function test5()
    # converting MNIST data for test inputs
	x_test, y_test = MNIST(split=:test)[:]
	x_test_flatten = flatten(x_test)

    mnist_model = create_MNIST_nn(
        Chain(
            Dense(784, 32, relu),
            Dense(32, 16, relu),
            Dense(16, 10)
        )
    )
    mnist_model_len = length(mnist_model)
    jm = create_JuMP_model(mnist_model, "predict")

    len = 1
    inputs = [x_test_flatten[:,i] for i in 1:len] # first len test digits
    for i in 1:len
        evaluate(jm, inputs[i])
        optimize!(jm)
        @assert has_values(jm) "JuMP model infeasible with input $(inputs[i])."

        nn_result = mnist_model(inputs[i])[1]
        JuMP_model_result = objective_value(jm)
        println("nn_result: ", nn_result)
        println("JuMP_model_result: ", JuMP_model_result)

        @assert abs(nn_result - JuMP_model_result) < 1e-3 "nn and JuMP model give different results."

        # prints the JuMP model predicted digit
        digit_guess = -1
        biggest_activation = -Inf
        for digit in 0:9
            activation = value(jm[:x][mnist_model_len, digit+1])
            if activation > biggest_activation
                biggest_activation = activation
                digit_guess = digit
            end
        end
        @assert 0 <= digit_guess <= 9 "No digit prediction was made"
        println("Predicted digit: ", digit_guess, ". Actual digit: ", y_test[i])
    end

    println("Test5 passed")
end

function unit_tests()
    test1()
    test2()
    test3()
    test4()
    test5()
    println("Unit tests passed")
end