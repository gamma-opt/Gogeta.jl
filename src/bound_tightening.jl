# solves the optimal BT bounds
# nn_model is the trained nn, bound_U and bounds_L are the initial bounds as of the algorithm
function solve_optimal_bounds(DNN::Chain, init_U_bounds::Vector{Float32}, init_L_bounds::Vector{Float32})

    K = length(DNN) # NOTE! there are K+1 layers in the nn
    DNN_params = params(DNN)

    W = [DNN_params[2*i-1] for i in 1:K] # weights of the i:th layer are stored at the (2*i-1):th index
    b = [DNN_params[2*i] for i in 1:K] # biases  of the i:th layer are stored at the (2*i):th index

    # stores the node count of layer k (starting at layer k=0) at index k+1
    input_node_count = length(DNN_params[1][1, :])
    node_count = [if k == 1 input_node_count else length(DNN_params[2*(k-1)]) end for k in 1:K+1]

    # store the current optimal bounds in the algorithm
    curr_L_bounds = copy(init_L_bounds)
    curr_U_bounds = copy(init_U_bounds)

	# Threads.@threads for obj_function in 1:2

		model = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 1))
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
				fix(U[k, j], curr_U_bounds[index], force=true)
				fix(L[k, j], curr_L_bounds[index], force=true)
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
					println("Curr layer: ", k, " , curr node: ", node, " , curr obj: ", obj_function)
					optimize!(model)
					@assert termination_status(model) == OPTIMAL "Problem in layer $k (1:$K) and node $node is infeasible."
					optimal = objective_value(model)

					if obj_function == 1 # Min
						# push!(opt_L, optimal)
						curr_L_bounds[input_node_count + outer_index] = optimal
						fix(L[k, node], optimal)
					elseif obj_function == 2 # Max
						# push!(opt_U, optimal)
						curr_U_bounds[input_node_count + outer_index] = optimal
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
	# end

    return curr_L_bounds, curr_U_bounds
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

bad_U1 = Float32[if i <= 784 1 else 1000 end for i in 1:842]
bad_L1 = Float32[if i <= 784 0 else -1000 end for i in 1:842]
@time optimal_L1, optimal_U1 = solve_optimal_bounds(nn1, bad_U1, bad_L1)

bad_U2 = Float32[if i <= 784 1 else 1000 end for i in 1:874]
bad_L2 = Float32[if i <= 784 0 else -1000 end for i in 1:874]
@time optimal_L2, optimal_U2 = solve_optimal_bounds(nn2, bad_U2, bad_L2)

bad_U3 = Float32[if i <= 784 1 else 1000 end for i in 1:890]
bad_L3 = Float32[if i <= 784 0 else -1000 end for i in 1:890]
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

test_times1, test_imgs1 = create_adversarials(nn1, optimal_U1, optimal_L1, 1, 10)