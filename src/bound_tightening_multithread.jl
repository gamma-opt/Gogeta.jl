# bound tightening but using workers

@everywhere function solve_optimal_bounds_multi(DNN::Chain, init_U_bounds::Vector{Float32}, init_L_bounds::Vector{Float32}, verbose::Bool=false)

    K = length(DNN) # NOTE! there are K+1 layers in the nn

    # store the DNN weights and biases
    DNN_params = params(DNN)
    W = [DNN_params[2*i-1] for i in 1:K]
    b = [DNN_params[2*i] for i in 1:K]

    # stores the node count of layer k (starting at layer k=0) at index k+1
    input_node_count = length(DNN_params[1][1, :])
    node_count = [if k == 1 input_node_count else length(DNN_params[2*(k-1)]) end for k in 1:K+1]

    # store the current optimal bounds in the algorithm
    curr_U_bounds = copy(init_U_bounds)
    curr_L_bounds = copy(init_L_bounds)

    # copy bounds to shared array
    shared_U_bounds = SharedArray(curr_U_bounds)
    shared_L_bounds = SharedArray(curr_L_bounds)
    
    for k in 1:K

        @sync @distributed for node in 1:(2*node_count[k+1]) # loop over both obj functions

            ### below variables and constraints in all problems

            model = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => (verbose ? 1 : 0), "Threads" => 1))

            # keeps track of the current node index starting from layer 1 (out of 0:K)
            prev_layers_node_sum = 0
            for prev_layer in 0:k-1
                prev_layers_node_sum += node_count[prev_layer+1]
            end
            
            # loops nodes twice: 1st time with obj function Min, 2nd time with Max
            curr_node = node
            obj_function = 1
            if node > node_count[k+1]
                curr_node = node - node_count[k+1]
                obj_function = 2
            end
            curr_node_index = prev_layers_node_sum + curr_node

            # NOTE! below variables and constraints for all opt problems
            @variable(model, x[k in 0:K, j in 1:node_count[k+1]] >= 0)
            @variable(model, s[k in 1:K-1, j in 1:node_count[k+1]] >= 0)
            @variable(model, z[k in 1:K-1, j in 1:node_count[k+1]], Bin)
            @variable(model, U[k in 0:K, j in 1:node_count[k+1]])
            @variable(model, L[k in 0:K, j in 1:node_count[k+1]])

            # fix values to all U[k,j] and L[k,j] from U_bounds and L_bounds
            index = 1
            for k in 0:K
                for j in 1:node_count[k+1]
                    fix(U[k, j], shared_U_bounds[index], force=true)
                    fix(L[k, j], shared_L_bounds[index], force=true)
                    index += 1
                end
            end

            # input layer (layer 0) node bounds are given beforehand
            for input_node in 1:node_count[1]
                delete_lower_bound(x[0, input_node])
                @constraint(model, L[0, input_node] <= x[0, input_node])
                @constraint(model, x[0, input_node] <= U[0, input_node])
            end

            # deleting lower bound for output nodes
            for output_node in 1:node_count[K+1]
                delete_lower_bound(x[K, output_node])
            end

            ### below constraints depending on the layer (every constraint up to the previous layer)
            for k_in in 1:k
                for node_in in 1:node_count[k_in]
                    if k_in >= 2
                        temp_sum = sum(W[k_in-1][node_in, j] * x[k_in-1-1, j] for j in 1:node_count[k_in-1])
                        @constraint(model, x[k_in-1, node_in] <= U[k_in-1, node_in] * z[k_in-1, node_in])
                        @constraint(model, s[k_in-1, node_in] <= -L[k_in-1, node_in] * (1 - z[k_in-1, node_in]))
                        if k_in <= K - 1
                            @constraint(model, temp_sum + b[k_in-1][node_in] == x[k_in-1, node_in] - s[k_in-1, node_in])
                        else # k_in == K
                            @constraint(model, temp_sum + b[k_in-1][node_in] == x[k_in-1, node_in])
                        end
                    end
                end
            end

            ### below constraints depending on the node
            temp_sum = sum(W[k][curr_node, j] * x[k-1, j] for j in 1:node_count[k]) # NOTE! prev layer [k]
            if k <= K - 1
                @constraint(model, node_con, temp_sum + b[k][curr_node] == x[k, curr_node] - s[k, curr_node])
                @constraint(model, node_U, x[k, curr_node] <= U[k, curr_node] * z[k, curr_node])
                @constraint(model, node_L, s[k, curr_node] <= -L[k, curr_node] * (1 - z[k, curr_node]))
            elseif k == K # == last value of k
                @constraint(model, node_con, temp_sum + b[k][curr_node] == x[k, curr_node])
                @constraint(model, node_L, L[k, curr_node] <= x[k, curr_node])
                @constraint(model, node_U, x[k, curr_node] <= U[k, curr_node])
            end

            # for obj_function in 1:2
                if obj_function == 1 && k <= K - 1 # Min, hidden layer
                    @objective(model, Min, x[k, curr_node] - s[k, curr_node])
                elseif obj_function == 2 && k <= K - 1 # Max, hidden layer
                    @objective(model, Max, x[k, curr_node] - s[k, curr_node])
                elseif obj_function == 1 && k == K # Min, last layer
                    @objective(model, Min, x[k, curr_node])
                elseif obj_function == 2 && k == K # Max, last layer
                    @objective(model, Max, x[k, curr_node])
                end

                solve_time = @elapsed optimize!(model)
                solve_time = round(solve_time; sigdigits = 3)
                @assert termination_status(model) == OPTIMAL 
                    "Problem (layer $k (from 1:$K), node $curr_node, $(obj_function == 1 ? "L" : "U")-bound) is infeasible."
                println("Solve time (layer $k, node $curr_node, $(obj_function == 1 ? "L" : "U")-bound): $(solve_time)s")
                optimal = objective_value(model)
                println("thread: ", myid(), ", node: ", curr_node, ", optimal value: ", optimal)

                # fix the model variable L or U corresponding to the current node to be the optimal value
                if obj_function == 1 # Min
                    shared_L_bounds[curr_node_index] = optimal
                    fix(L[k, curr_node], optimal)
                elseif obj_function == 2 # Max
                    shared_U_bounds[curr_node_index] = optimal
                    fix(U[k, curr_node], optimal)
                end
            # end
            
        end

    end

    println("Solving optimal constraint bounds complete")

    len = length(curr_L_bounds)
    for i in 1:len
        curr_U_bounds[i] = shared_U_bounds[i]
        curr_L_bounds[i] = shared_L_bounds[i]
    end

    return curr_U_bounds, curr_L_bounds
end