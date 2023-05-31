
# creates a JuMP model for arbitrary sized / shaped nn
# type: what kind of model. adversial_index: index in train set to create new image
function create_JuMP_model(nn_model, bounds_U, bounds_L, type, adversial_index=-1, opt_img_digit=0)
    @assert type == "predict" ||
            type == "missclassified L1" ||
            type == "missclassified L2"
    "Invalid type attribute \"$type\""
    @assert 0 <= opt_img_digit <= 9 "Digit must be between 0 and 9."

    K = length(nn_model) # NOTE! there are K+1 layers in the nn
    nn_parameters = params(nn_model)
    # println("nn_parameters: $nn_parameters")

    W = [nn_parameters[2*i-1] for i in 1:K] # weights of the i:th layer are stored at the (2*i-1):th index
    b = [nn_parameters[2*i] for i in 1:K] # biases  of the i:th layer are stored at the (2*i):th index

    # stores the node count of layer k (starting at layer k=0) at index k+1
    input_node_count = length(nn_parameters[1][1, :])
    node_count = [
        if k == 1
            input_node_count
        else
            length(nn_parameters[2*(k-1)])
        end for k in 1:K+1
    ]

    model = Model(Gurobi.Optimizer)

    # sets the variables x[k,j] and s[k,j], the binary variables z[k,j] and the big-M values U[k,j] and L[k,j]
    @variable(model, x[k in 0:K, j in 1:node_count[k+1]] >= 0)
    @variable(model, s[k in 1:K, j in 1:node_count[k+1]] >= 0)
    @variable(model, z[k in 1:K, j in 1:node_count[k+1]], Bin)
    @variable(model, U[k in 0:K, j in 1:node_count[k+1]])
    @variable(model, L[k in 0:K, j in 1:node_count[k+1]])

    # fix bounds to all U[k,j] and L[k,j] from bounds_U and bounds_L
    index = 1
    for k in 0:K
        for j in 1:node_count[k+1]
            fix(U[k, j], bounds_U[index])
            fix(L[k, j], bounds_L[index])
            index += 1
        end
    end

    # constraint (4a)
    for input_node in 1:node_count[1]
        delete_lower_bound(x[0, input_node])
        @constraint(model, L[0, input_node] <= x[0, input_node])
        @constraint(model, x[0, input_node] <= U[0, input_node])
    end

    # constraint (4b) and (4e) (cases k=1, ..., k=K)
    for k in 1:K
        for node in 1:node_count[k+1] # node count of the next layer of k, i.e., the layer k+1 
            if k == 1 # constraint (4b), case k=1
                temp_sum = sum(W[k][node, j] * x[k-1, j] for j in 1:node_count[k])
                @constraint(model, temp_sum + b[k][node] == x[k, node] - s[k, node])
            else
                # [j in node_count[k]] is the number of nodes of the layer k-1 (although indexed k)
                temp_sum = sum(W[k][node, j] * x[k-1, j] for j in 1:node_count[k])
                if k < K # constraint (4b), case k=2, ..., k=K-1
                    @constraint(model, temp_sum + b[k][node] == x[k, node] - s[k, node])
                elseif k == K # constraint (4e) (k=K)
                    @constraint(model, temp_sum + b[k][node] == x[k, node])
                end
            end
        end
    end

    # constraint (4d)
    @constraint(model, [k in 1:K, j in 1:node_count[k+1]], x[k, j] <= U[k, j] * z[k, j])
    @constraint(model, [k in 1:K, j in 1:node_count[k+1]], s[k, j] <= -L[k, j] * (1 - z[k, j]))

    # constraint (4f)
    for output_node in 1:node_count[K+1]
        delete_lower_bound(x[K, output_node])
        @constraint(model, L[K, output_node] <= x[K, output_node])
        @constraint(model, x[K, output_node] <= U[K, output_node])
    end

    # NOTE! Below if clauses for testing, the type attribute will determine which objective
    # function will be added to the model, as well as if other variables or constraints
    # need to be added (e.g., (12) and (13) from the paper for type "missclassified")

    x_train, y_train = MNIST(split=:train)[:] # easy access for testing
    x_train_flatten = flatten(x_train)

    if type == "predict"

        @objective(model, Max, x[K, opt_img_digit+1]) # objective function for testing

    elseif type == "missclassified L1"

        @assert adversial_index >= 1 "adversial_index missing from input"
        cur_digit = y_train[adversial_index]
        cur_digit_img = x_train_flatten[:, adversial_index]

        # variables for constraint (12) in the 2018 paper
        @variable(model, d[k in [0], j in 1:node_count[1]] >= 0)

        mult = 1.2
        imposed_index = (cur_digit + 5) % 10 + 1 # digit is imposed as (d + 5 mod 10), +1 for indexing
        for output_node in 1:node_count[K+1]
            if output_node != imposed_index
                @constraint(model, x[K, imposed_index] >= mult * x[K, output_node])
            end
        end
        # (13) in the paper
        for input_node in 1:node_count[1]
            @constraint(model, -d[0, input_node] <= x[0, input_node] - cur_digit_img[input_node])
            @constraint(model, x[0, input_node] - cur_digit_img[input_node] <= d[0, input_node])
        end

        # the obj function from the paper
        @objective(model, Min, sum(d[0, input_node] for input_node in 1:node_count[1]))

    elseif type == "missclassified L2"

        @assert adversial_index >= 1 "adversial_index missing from input"
        cur_digit = y_train[adversial_index]
        cur_digit_img = x_train_flatten[:, adversial_index]

        mult = 1.2
        imposed_index = (cur_digit + 5) % 10 + 1 # digit is imposed as (d + 5 mod 10), +1 for indexing
        for output_node in 1:node_count[K+1]
            if output_node != imposed_index
                @constraint(model, x[K, imposed_index] >= mult * x[K, output_node])
            end
        end

        # the obj function from the paper
        @objective(model, Min, sum((x[0, input_node] - cur_digit_img[input_node])^2 for input_node in 1:node_count[1]))
    end

    # println(model)

    return model
end

# fixes the input values (layer k=0) for the JuMP model
function evaluate(JuMP_model, input)

    x = JuMP_model[:x] # stores the @variable with name x from the JuMP model
    input_len = length(input)
    for input_node in 1:input_len
        fix(x[0, input_node], input[input_node], force=true) # fix value of input to x[0,j]
    end
    # println(JuMP_model)
end


# solves the optimal BT bounds
# nn_model is the trained nn, bound_U and bounds_L are the initial bounds as of the algorithm
function solve_optimal_bounds(nn_model, bounds_U, bounds_L)

    K = length(nn_model) # NOTE! there are K+1 layers in the nn
    nn_parameters = params(nn_model)
    # println("nn_parameters: $nn_parameters")

    W = [nn_parameters[2*i-1] for i in 1:K] # weights of the i:th layer are stored at the (2*i-1):th index
    b = [nn_parameters[2*i] for i in 1:K] # biases  of the i:th layer are stored at the (2*i):th index

    # stores the node count of layer k (starting at layer k=0) at index k+1
    input_node_count = length(nn_parameters[1][1, :])
    node_count = [
        if k == 1
            input_node_count
        else
            length(nn_parameters[2*(k-1)])
        end for k in 1:K+1
    ]

    # store the current optimal bounds in the algorithm
    curr_bounds_U = copy(bounds_U)
    curr_bounds_L = copy(bounds_L)

    # these store the optimization models to determine the optimal U and L
    opt_L = []
    opt_U = []

    model = Model(Gurobi.Optimizer)
    outer_index = 1

    # NOTE! below variables and constraints for all opt problems
    @variable(model, x[k in 0:K, j in 1:node_count[k+1]] >= 0)
    @variable(model, s[k in 1:K-1, j in 1:node_count[k+1]] >= 0)
    @variable(model, 0 <= z[k in 1:K-1, j in 1:node_count[k+1]] <= 1)
    @variable(model, U[k in 0:K, j in 1:node_count[k+1]])
    @variable(model, L[k in 0:K, j in 1:node_count[k+1]])

    # fix bounds to all U[k,j] and L[k,j] from bounds_U and bounds_L
    index = 1
    for k in 0:K
        for j in 1:node_count[k+1]
            fix(U[k, j], curr_bounds_U[index], force=true)
            fix(L[k, j], curr_bounds_L[index], force=true)
            index += 1
        end
    end

    for input_node in 1:node_count[1] # input constraints (4a)
        delete_lower_bound(x[0, input_node])
        @constraint(model, L[0, input_node] <= x[0, input_node])
        @constraint(model, x[0, input_node] <= U[0, input_node])
    end

    for output_node in 1:node_count[K+1]
        delete_lower_bound(x[K, output_node])
        @constraint(model, L[K, output_node] <= x[K, output_node])
        @constraint(model, x[K, output_node] <= U[K, output_node])
    end


    for k in 1:K
        # NOTE! below constraints depending on the layer
        # we only want to build ALL of the constraints until the PREVIOUS layer, and then go node by node. Here we calculate ONLY the constraints until the PREVIOUS layer
        for node_in in 1:node_count[k]
            if k >= 2
                temp_sum = sum(W[k-1][node_in, j] * x[k-1-1, j] for j in 1:node_count[k-1])
                @constraint(model, x[k-1, node_in] <= U[k-1, node_in] * z[k-1, node_in])
                @constraint(model, s[k-1, node_in] <= -L[k-1, node_in] * (1 - z[k-1, node_in]))

                if k <= K - 1
                    @constraint(model, temp_sum + b[k-1][node_in] == x[k-1, node_in] - s[k-1, node_in])
                else # k == K
                    @constraint(model, temp_sum + b[k-1][node_in] == x[k-1, node_in])
                end
            end
        end

        for node in 1:node_count[k+1]
            # NOTE! below constraints depending on the node
            # Here we calculate the specific constraints depending on the NODE
            temp_sum = sum(W[k][node, j] * x[k-1, j] for j in 1:node_count[k]) # NOTE! prev layer [k]

            if k <= K - 1
                @constraint(model, node_con, temp_sum + b[k][node] == x[k, node] - s[k, node])
                @constraint(model, node_U, x[k, node] <= U[k, node] * z[k, node])
                @constraint(model, node_L, s[k, node] <= -L[k, node] * (1 - z[k, node]))
            elseif k == K # == last value of k
                @constraint(model, node_con, temp_sum + b[k][node] == x[k, node])
                @constraint(model, node_L, L[k, node] <= x[k, node]) # const (4f) in layer K
                @constraint(model, node_U, x[k, node] <= U[k, node])
            end

            for obj_function in 1:2
                if obj_function == 1 && k <= K - 1 # Min, hidden layer
                    @objective(model, Min, x[k, node] - s[k, node])
                elseif obj_function == 2 && k <= K - 1 # Max, hidden layer
                    @objective(model, Max, x[k, node] - s[k, node])
                elseif obj_function == 1 && k == K # Min, last layer
                    @objective(model, Min, x[k, node])
                elseif obj_function == 2 && k == K # Max, last layer
                    @objective(model, Max, x[k, node])
                end

                optimize!(model)
                @assert termination_status(model) == OPTIMAL "Problem in layer $k (1:$K) and node $node is infeasible."
                optimal = objective_value(model)

                if obj_function == 1 # Min
                    push!(opt_L, optimal)
                    curr_bounds_L[784+outer_index] = optimal
                    fix(L[k, node], optimal)
                elseif obj_function == 2 # Max
                    push!(opt_U, optimal)
                    curr_bounds_U[784+outer_index] = optimal
                    fix(U[k, node], optimal)
                end
            end
            outer_index += 1

            # deleting and unregistering the constraints assigned to the current node
            delete(model, node_con)
            delete(model, node_L)
            delete(model, node_U)
            unregister(model, :node_con)
            unregister(model, :node_L)
            unregister(model, :node_U)
        end
    end

    return opt_L, opt_U
end

# sum_model2 = create_sum_nn(
#         100, 5, 2, Chain(
#                    Dense(2, 4, relu),
#                    Dense(4, 3, relu),
#                    Dense(3, 1)), 
# )
# sum_U = [1,1,1000,1000,1000,1000,1000,1000,1000,1000]
# sum_L = [0,0,-1000,-1000,-1000,-1000,-1000,-1000,-1000,-1000]
# mdl2_min, mdl2_max = solve_optimal_bounds(sum_model2, sum_U, sum_L)
# opt2_min = solve_bounds(mdl2_min)
# opt2_max = solve_bounds(mdl2_max)

x_train, y_train = MNIST(split=:train)[:]
x_test, y_test = MNIST(split=:test)[:]
function test_nns(seed)
    Random.seed!(seed)
    nn1 = Chain( # 842 nodes
        Dense(784, 32, relu),
        Dense(32, 16, relu),
        Dense(16, 10)
    )
    nn2 = Chain( # 874 nodes
        Dense(784, 32, relu),
        Dense(32, 32, relu),
        Dense(32, 16, relu),
        Dense(16, 10)
    )
    nn3 = Chain( # 854 nodes
        Dense(784, 32, relu),
        Dense(32, 32, relu),
        Dense(32, 16, relu),
        Dense(16, 16, relu),
        Dense(16, 10)
    )
    return nn1, nn2, nn3
end

raw_nn1, raw_nn2, raw_nn3 = test_nns(2)

nn1, acc1 = train_mnist_nn(raw_nn1)
nn2, acc2 = train_mnist_nn(raw_nn2)
nn3, acc3 = train_mnist_nn(raw_nn3)

bad_U1 = Float32[
    if i <= 784
        1
    else
        1000
    end for i in 1:842
]
bad_L1 = Float32[
    if i <= 784
        0
    else
        -1000
    end for i in 1:842
]
@time optimal_L1, optimal_U1 = solve_optimal_bounds(nn1, bad_U1, bad_L1)
good_U1 = Float32[
    if i <= 784
        1
    else
        optimal_U1[i-784]
    end for i in 1:842
]
good_L1 = Float32[
    if i <= 784
        0
    else
        optimal_L1[i-784]
    end for i in 1:842
]

bad_U2 = Float32[
    if i <= 784
        1
    else
        1000
    end for i in 1:874
]
bad_L2 = Float32[
    if i <= 784
        0
    else
        -1000
    end for i in 1:874
]
@time optimal_L2, optimal_U2 = solve_optimal_bounds(nn2, bad_U2, bad_L2)
good_U2 = Float32[
    if i <= 784
        1
    else
        optimal_U2[i-784]
    end for i in 1:874
]
good_L2 = Float32[
    if i <= 784
        0
    else
        optimal_L2[i-784]
    end for i in 1:874
]

bad_U3 = Float32[
    if i <= 784
        1
    else
        1000
    end for i in 1:890
]
bad_L3 = Float32[
    if i <= 784
        0
    else
        -1000
    end for i in 1:890
]
@time optimal_L3, optimal_U3 = solve_optimal_bounds(nn3, bad_U3, bad_L3)
good_U3 = Float32[
    if i <= 784
        1
    else
        optimal_U3[i-784]
    end for i in 1:890
]
good_L3 = Float32[
    if i <= 784
        0
    else
        optimal_L3[i-784]
    end for i in 1:890
]


function opt_times(nn, U, L, range)
    times = []
    images = []
    for i in range
        println("Current index is $i")
        mdl = create_JuMP_model(nn, U, L, "missclassified L1", i)
        time = @elapsed optimize!(mdl)
        adversarial = Float32[]
        for pixel in 1:784
            pixel_value = value(mdl[:x][0, pixel]) # indexing
            push!(adversarial, pixel_value)
        end
        push!(images, adversarial)
        push!(times, time)
    end
    return times, images
end

bad_times1, bad_imgs1 = opt_times(nn1, bad_U1, bad_L1, 1:10)
good_times1, good_imgs1 = opt_times(nn1, good_U1, good_L1, 1:10)
difference1 = bad_times1 - good_times1

bad_times2, bad_imgs2 = opt_times(nn2, bad_U2, bad_L2, 1:10)
good_times2, good_imgs2 = opt_times(nn2, good_U2, good_L2, 1:10)
difference2 = bad_times2 - good_times2

bad_times3, bad_imgs3 = opt_times(nn3, bad_U3, bad_L3, 1:10)
good_times3, good_imgs3 = opt_times(nn3, good_U3, good_L3, 1:10)
difference3 = bad_times3 - good_times3

test_times1, test_imgs1 = opt_times(nn1, good_U1, good_L1, 1:10)
