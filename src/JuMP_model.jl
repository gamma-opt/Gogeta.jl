
# creates a JuMP model for arbitrary sized / shaped nn
# type: what kind of model. adversial_index: index in train set to create new image
function create_JuMP_model(nn_model, type, adversial_index = -1, opt_img_digit = 0)
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
	input_node_count = length(nn_parameters[1][1,:])
	node_count = [if k == 1 input_node_count else length(nn_parameters[2*(k-1)]) end for k in 1:K+1]

	model = Model(Gurobi.Optimizer)
	
	# sets the variables x[k,j] and s[k,j], the binary variables z[k,j] and the big-M values U[k,j] and L[k,j]
	@variable(model, x[k in 0:K, j in 1:node_count[k+1]] >= 0)
	@variable(model, s[k in 1:K, j in 1:node_count[k+1]] >= 0)
	@variable(model, z[k in 1:K, j in 1:node_count[k+1]], Bin)
	# big-M values (currently) fixed for all U[k,j] and L[k,j], input fixed to [0,1] for image pixel values
	big_M = Dict(:"U" => 1000, :"L" => -1000, "U_input" => 1, "L_input" => 0) 

	# constraint (4a)
	for input_node in 1:node_count[1]
		delete_lower_bound(x[0, input_node])
		@constraint(model, big_M["L_input"] <= x[0, input_node])
		@constraint(model, x[0, input_node] <= big_M["U_input"])
	end

	# constraint (4b) and (4e) (cases k=1, ..., k=K)
	for k in 1:K
		for node in 1:node_count[k+1] # node count of the next layer of k, i.e., the layer k+1 
			if k == 1 # constraint (4b), case k=1
				temp_sum = sum(W[k][node,j] * x[k-1,j] for j in 1:node_count[k])
				@constraint(model, temp_sum + b[k][node] == x[k,node] - s[k,node])
			else
				# [j in node_count[k]] is the number of nodes of the layer k-1 (although indexed k)
				temp_sum = sum(W[k][node,j] * x[k-1,j] for j in 1:node_count[k])
				if k < K # constraint (4b), case k=2, ..., k=K-1
					@constraint(model, temp_sum + b[k][node] == x[k,node] - s[k,node])
				elseif k == K # constraint (4e) (k=K)
					@constraint(model, temp_sum + b[k][node] == x[k,node])
				end
			end
		end
	end

	# constraint (4d)
	@constraint(model, [k in 1:K, j in 1:node_count[k+1]], x[k,j] <= big_M["U"] * z[k,j]) 
	@constraint(model, [k in 1:K, j in 1:node_count[k+1]], s[k,j] <= -big_M["L"] * (1 - z[k,j]))

	# constraint (4f)
	for output_node in 1:node_count[K+1]
		delete_lower_bound(x[K, output_node])
		@constraint(model, big_M["L"] <= x[K, output_node])
		@constraint(model, x[K, output_node] <= big_M["U"])
	end

	# NOTE! Below if clauses for testing, the type attribute will determine which objective
	# function will be added to the model, as well as if other variables or constraints
	# need to be added (e.g., (12) and (13) from the paper for type "missclassified")

	x_train, y_train = MNIST(split=:train)[:] # easy access for testing
	x_train_flatten = flatten(x_train)

	if type == "predict"

		@objective(model, Max, x[K,opt_img_digit + 1]) # objective function for testing

	elseif type == "missclassified L1"

		@assert adversial_index >= 1 "adversial_index missing from input"
		cur_digit = y_train[adversial_index]
		cur_digit_img = x_train_flatten[:,adversial_index]

		# variables for constraint (12) in the 2018 paper
		@variable(model, d[k in [0], j in 1:node_count[1]] >= 0) 

		mult = 1.2
		imposed_index = (cur_digit + 5) % 10 + 1 # digit is imposed as (d + 5 mod 10), +1 for indexing
		for output_node in 1:node_count[K+1]
			if output_node != imposed_index
				@constraint(model, x[K,imposed_index] >= mult * x[K,output_node])
			end
		end
		# (13) in the paper
		for input_node in 1:node_count[1]
			@constraint(model, -d[0,input_node] <= x[0,input_node] - cur_digit_img[input_node])
			@constraint(model, x[0,input_node] - cur_digit_img[input_node] <= d[0,input_node])
		end

		# the obj function from the paper
		@objective(model, Min, sum(d[0,input_node] for input_node in 1:node_count[1]))

	elseif type == "missclassified L2"

		@assert adversial_index >= 1 "adversial_index missing from input"
		cur_digit = y_train[adversial_index]
		cur_digit_img = x_train_flatten[:,adversial_index]

		mult = 1.2
		imposed_index = (cur_digit + 5) % 10 + 1 # digit is imposed as (d + 5 mod 10), +1 for indexing
		for output_node in 1:node_count[K+1]
			if output_node != imposed_index
				@constraint(model, x[K,imposed_index] >= mult * x[K,output_node])
			end
		end

		# the obj function from the paper
		@objective(model, Min, sum((x[0,input_node] - cur_digit_img[input_node])^2 for input_node in 1:node_count[1]))
	end

	# println(model)

	return model
end

# fixes the input values (layer k=0) for the JuMP model
function evaluate(JuMP_model, input) 

    x = JuMP_model[:x] # stores the @variable with name x from the JuMP model
    input_len = length(input)
    for input_node in 1:input_len
        fix(x[0, input_node], input[input_node], force = true) # fix value of input to x[0,j]
    end
    # println(JuMP_model)
end


# solves the optimal BT bounds
function solve_optimal_bounds(nn_model, input_U, input_L, type = "predict")
	@assert type == "predict" "Invalid type attribute \"$type\""

	K = length(nn_model) # NOTE! there are K+1 layers in the nn
	nn_parameters = params(nn_model)
	# println("nn_parameters: $nn_parameters")

	W = [nn_parameters[2*i-1] for i in 1:K] # weights of the i:th layer are stored at the (2*i-1):th index
	b = [nn_parameters[2*i] for i in 1:K] # biases  of the i:th layer are stored at the (2*i):th index
	
	# stores the node count of layer k (starting at layer k=0) at index k+1
	input_node_count = length(nn_parameters[1][1,:])
	node_count = [if k == 1 input_node_count else length(nn_parameters[2*(k-1)]) end for k in 1:K+1]

	# these store the optimization models to determine the optimal U and L
	min_models = []
	max_models = []
	for k in 1:K
		for node in 1:node_count[k+1]
			println("k: ", k, " j: ", node)
		end
	end

	for k in 1:K
		for node in 1:node_count[k+1]

			model = Model(Gurobi.Optimizer)
			# NOTE! below constraints in every problem

			# sets the variables x[k,j] and s[k,j], the relaxed z[k,j] and the big-M values U[k,j] and L[k,j]
			@variable(model, x[k in 0:K, j in 1:node_count[k+1]] >= 0)
			@variable(model, s[k in 1:K, j in 1:node_count[k+1]] >= 0)
			@variable(model, 0 <= z[k in 1:K, j in 1:node_count[k+1]] <= 1)
			@variable(model, U[k in 0:K, j in 1:node_count[k+1]] == 1000)
			@variable(model, L[k in 0:K, j in 1:node_count[k+1]] == -1000)
			# fix input bounds from input_U and input_L
			for j in 1:node_count[1] 
				fix(U[0,j], input_U[j])
				fix(L[0,j], input_L[j])
			end

			for input_node in 1:node_count[1] # input constraints
				delete_lower_bound(x[0, input_node])
				@constraint(model, L[0, input_node] <= x[0, input_node])
				@constraint(model, x[0, input_node] <= U[0, input_node])
			end

			# NOTE! below constraints depending on the layer
			# we only want to build ALL of the constraints until the PREVIOUS layer, and then go node by node
			for k_in in 0:k-1
				if k_in >= 1
					temp_sum = sum(W[k_in][node,j] * x[k_in-1,j] for j in 1:node_count[k_in]) # NOTE! prev layer [k_in]
					@constraint(model, x[k_in,node] <= U[k_in,node] * z[k_in,node])
					@constraint(model, s[k_in,node] <= -L[k_in,node] * (1 - z[k_in,node]))

					if k_in <= K-1-1
						@constraint(model, temp_sum + b[k_in][node] == x[k_in,node] - s[k_in,node])
					else # k_in == K-1
						@constraint(model, temp_sum + b[k_in][node] == x[k_in,node])
					end
				end
			end

			# NOTE! below constraints depending on the node
			temp_sum = sum(W[k][node,j] * x[k-1,j] for j in 1:node_count[k]) # NOTE! prev layer [k]
			@constraint(model, x[k,node] <= U[k,node] * z[k,node])
			@constraint(model, s[k,node] <= -L[k,node] * (1 - z[k,node]))

			if k <= K-1
				@constraint(model, temp_sum + b[k][node] == x[k,node] - s[k,node])
				@objective(model, Max, x[k,node] - s[k,node])
			else # k == K
				@constraint(model, temp_sum + b[k][node] == x[k,node])
				@objective(model, Max, x[k,node])
			end

			println(model)
			push!(max_models, model)
		end
	end

	# # constraint (4b) and (4e) (cases k=1, ..., k=K)
	# for k in 1:K
	# 	for node in 1:node_count[k+1] # node count of the next layer of k, i.e., the layer k+1 
	# 		if k == 1 # constraint (4b), case k=1
	# 			temp_sum = sum(W[k][node,j] * x[k-1,j] for j in 1:node_count[k])
	# 			@constraint(model, temp_sum + b[k][node] == x[k,node] - s[k,node])
	# 		else
	# 			# [j in node_count[k]] is the number of nodes of the layer k-1 (although indexed k)
	# 			temp_sum = sum(W[k][node,j] * x[k-1,j] for j in 1:node_count[k])
	# 			if k < K # constraint (4b), case k=2, ..., k=K-1
	# 				@constraint(model, temp_sum + b[k][node] == x[k,node] - s[k,node])
	# 			elseif k == K # constraint (4e) (k=K)
	# 				@constraint(model, temp_sum + b[k][node] == x[k,node])
	# 			end
	# 		end
	# 	end
	# end

	# # constraint (4d)
	# @constraint(model, [k in 1:K, j in 1:node_count[k+1]], x[k,j] <= U[k,j] * z[k,j])
	# @constraint(model, [k in 1:K, j in 1:node_count[k+1]], s[k,j] <= -L[k,j] * (1 - z[k,j]))

	# # constraint (4f)
	# for output_node in 1:node_count[K+1]
	# 	delete_lower_bound(x[K, output_node])
	# 	@constraint(model, L[K, output_node] <= x[K, output_node])
	# 	@constraint(model, x[K, output_node] <= U[K, output_node])
	# end

	# if type == "predict"

	# 	@objective(model, Max, x[K,opt_img_digit + 1]) # objective function for testing

    # end

	return max_models
end
