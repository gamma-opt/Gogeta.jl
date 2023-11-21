using JuMP, Flux, Gurobi
using JuMP: Model
using Flux: params
using Distributed
using SharedArrays

"""
bound_tightening(DNN::Chain, init_U_bounds::Vector{Float32}, init_L_bounds::Vector{Float32}, verbose::Bool=false)

A single-threaded implementation of optimal tightened constraint bounds L and U for for a trained DNN.
Using these bounds with the create_JuMP_model function reduces solution time for optimization problems.

# Arguments
- `DNN::Chain`: A trained ReLU DNN.
- `init_U_bounds::Vector{Float32}`: Initial upper bounds on the node values of the DNN.
- `init_L_bounds::Vector{Float32}`: Initial lower bounds on the node values of the DNN.
- `verbose::Bool=false`: Controls Gurobi logs.

# Examples
```julia
L_bounds, U_bounds = bound_tightening(DNN, init_U_bounds, init_L_bounds, false)
```
"""

function bound_tightening(DNN::Chain, init_U_bounds::Vector{Float64}, init_L_bounds::Vector{Float64}, verbose::Bool=false)

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

    model = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => (verbose ? 1 : 0)))

    # keeps track of the current node index starting from layer 1 (out of 0:K)
    outer_index = node_count[1] + 1

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
            fix(U[k, j], curr_U_bounds[index], force=true)
            fix(L[k, j], curr_L_bounds[index], force=true)
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

    # NOTE! below constraints depending on the layer
    for k in 1:K
        println()
        println("Working on layer $k.")
        # we only want to build ALL of the constraints until the PREVIOUS layer, and then go node by node
        # here we calculate ONLY the constraints until the PREVIOUS layer
        for node_in in 1:node_count[k]
            if k >= 2
                temp_sum = AffExpr()
                for j in 1:node_count[k-1]
                    add_to_expression!(temp_sum, W[k-1][node_in, j], x[k-1-1, j])
                end
                # temp_sum = sum(W[k-1][node_in, j] * x[k-1-1, j] for j in 1:node_count[k-1])
                @constraint(model, x[k-1, node_in] <= U[k-1, node_in] * z[k-1, node_in])
                @constraint(model, s[k-1, node_in] <= -L[k-1, node_in] * (1 - z[k-1, node_in]))
                if k <= K - 1
                    @constraint(model, temp_sum + b[k-1][node_in] == x[k-1, node_in] - s[k-1, node_in])
                else # k == K
                    @constraint(model, temp_sum + b[k-1][node_in] == x[k-1, node_in])
                end
            end
        end

        # NOTE! below constraints depending on the node
        for node in 1:node_count[k+1]
            # here we calculate the specific constraints depending on the current node
            temp_sum = AffExpr()
            for j in 1:node_count[k]
                add_to_expression!(temp_sum, W[k][node, j], x[k-1, j])
            end
            if k <= K - 1
                @constraint(model, node_con, temp_sum + b[k][node] == x[k, node] - s[k, node])
                @constraint(model, node_U, x[k, node] <= U[k, node] * z[k, node])
                @constraint(model, node_L, s[k, node] <= -L[k, node] * (1 - z[k, node]))
            elseif k == K # == last value of k
                @constraint(model, node_con, temp_sum + b[k][node] == x[k, node])
                @constraint(model, node_L, L[k, node] <= x[k, node])
                @constraint(model, node_U, x[k, node] <= U[k, node])
            end

            # NOTE! below objective function and optimizing the model depending on obj_function and layer
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

                solve_time = @elapsed optimize!(model)
                solve_time = round(solve_time; sigdigits = 3)

                optimal = nothing

                try
                    optimal = objective_value(model)
                catch
                    optimal = obj_function == 1 ? curr_L_bounds[outer_index] : curr_U_bounds[outer_index]
                end

                print(".")

                # fix the model variable L or U corresponding to the current node to be the optimal value
                if obj_function == 1 # Min
                    curr_L_bounds[outer_index] = optimal
                    fix(L[k, node], optimal)
                elseif obj_function == 2 # Max
                    curr_U_bounds[outer_index] = optimal
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

    println("Solving optimal constraint bounds single-threaded complete")

    return curr_U_bounds, curr_L_bounds
end

function bound_tightening_threads_old(DNN::Chain, init_U_bounds::Vector{Float64}, init_L_bounds::Vector{Float64}, verbose::Bool=false)

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

    lock = Threads.ReentrantLock()

    for k in 1:K

        GC.gc()

        println()
        println("Working on layer $k.")

        Threads.@threads for node in 1:(2*node_count[k+1]) # loop over both obj functions

            ### below variables and constraints in all problems

            model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env)))
            # set_optimizer_attribute(model, "MIPGap", 0.1)
            set_silent(model)

            # model = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => (verbose ? 1 : 0)))

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
            Threads.lock(lock) do
                for k in 0:K
                    for j in 1:node_count[k+1]
                        fix(U[k, j], curr_U_bounds[index], force=true)
                        fix(L[k, j], curr_L_bounds[index], force=true)
                        index += 1
                    end
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

            optimal = nothing
            # @assert termination_status(model) == OPTIMAL || termination_status(model) == TIME_LIMIT
            #     "Problem (layer $k (from 1:$K), node $curr_node, $(obj_function == 1 ? "L" : "U")-bound) is infeasible."
            try
                optimal = objective_value(model)
            catch
                optimal = obj_function == 1 ? curr_L_bounds[curr_node_index] : curr_U_bounds[curr_node_index]
            end

            print(".")

            # fix the model variable L or U corresponding to the current node to be the optimal value
            Threads.lock(lock) do
                if obj_function == 1 # Min
                    curr_L_bounds[curr_node_index] = optimal
                    fix(L[k, curr_node], optimal)
                elseif obj_function == 2 # Max
                    curr_U_bounds[curr_node_index] = optimal
                    fix(U[k, curr_node], optimal)
                end
            end

            model = nothing

        end

    end

    println("Solving optimal constraint bounds using threads complete")

    return curr_U_bounds, curr_L_bounds
end

"""
bound_tightening_threads(DNN::Chain, init_U_bounds::Vector{Float32}, init_L_bounds::Vector{Float32}, verbose::Bool=false)

A multi-threaded (using Threads) implementation of optimal tightened constraint bounds L and U for for a trained DNN.
Using these bounds with the create_JuMP_model function reduces solution time for optimization problems.

# Arguments
- `DNN::Chain`: A trained ReLU DNN.
- `init_U_bounds::Vector{Float32}`: Initial upper bounds on the node values of the DNN.
- `init_L_bounds::Vector{Float32}`: Initial lower bounds on the node values of the DNN.
- `verbose::Bool=false`: Controls Gurobi logs.

# Examples
```julia
L_bounds_threads, U_bounds_threads = bound_tightening_threads(DNN, init_U_bounds, init_L_bounds, false)
```
"""

function bound_tightening_threads(DNN::Chain, init_U_bounds::Vector{Float64}, init_L_bounds::Vector{Float64}, verbose::Bool=false, gurobi_env=Gurobi.Env())

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

    lock = Threads.ReentrantLock()

    for k in 1:K
        prev_layers_node_sum = sum(node_count[1:k])

        println()
        println("Working on layer $k.")

        GC.gc()

        Threads.@threads for node in 1:(2*node_count[k+1]) # loop over both obj functions

            optimal, obj_function, curr_node_index = bound_tightening_threads_inner(
                k,
                K,
                node,
                node_count,
                prev_layers_node_sum,
                curr_U_bounds,
                curr_L_bounds,
                W,
                b,
                verbose,
                gurobi_env
            )

            # fix the model variable L or U corresponding to the current node to be the optimal value
            Threads.lock(lock) do
                if obj_function == 1 # Min
                    curr_L_bounds[curr_node_index] = optimal
                    # fix(L[k, curr_node], optimal)
                elseif obj_function == 2 # Max
                    curr_U_bounds[curr_node_index] = optimal
                    # fix(U[k, curr_node], optimal)
                end
            end

        end

    end

    println("Solving optimal constraint bounds using threads complete")

    return curr_U_bounds, curr_L_bounds
end

function bound_tightening_threads_inner(k, K, node, node_count, prev_layers_node_sum, current_upper_bounds, current_lower_bounds, W, b, verbose, gurobi_env)
    creation_time = @elapsed begin

    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env)))
    # model attributes
    set_attribute(model, "OutputFlag", (verbose ? 1 : 0))
    set_attribute(model, "TimeLimit", 10)
    # set_optimizer_attribute(model, "MIPGap", 0.1)
    set_silent(model)

    # loops nodes twice: 1st time with obj function Min, 2nd time with Max
    curr_node = node
    obj_function = 1
    if node > node_count[k + 1]
        curr_node = node - node_count[k + 1]
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
        for j in 1:node_count[k + 1]
            fix(U[k, j], current_upper_bounds[index], force=true)
            fix(L[k, j], current_lower_bounds[index], force=true)
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
                temp_sum = AffExpr()
                for j in 1:node_count[k_in-1]
                    add_to_expression!(temp_sum, W[k_in-1][node_in, j], x[k_in - 2, j])
                end
                # temp_sum = sum(W[k_in-1][node_in, j] * x[k_in-1-1, j] for j in 1:node_count[k_in-1])
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
    temp_sum = AffExpr()
    for j in 1:node_count[k]
        add_to_expression!(temp_sum, W[k][curr_node, j], x[k - 1, j])
    end

    if k <= K - 1
        @constraint(model, node_con, temp_sum + b[k][curr_node] == x[k, curr_node] - s[k, curr_node])
        @constraint(model, node_U, x[k, curr_node] <= U[k, curr_node] * z[k, curr_node])
        @constraint(model, node_L, s[k, curr_node] <= -L[k, curr_node] * (1 - z[k, curr_node]))
    elseif k == K # == last value of k
        @constraint(model, node_con, temp_sum + b[k][curr_node] == x[k, curr_node])
        @constraint(model, node_L, L[k, curr_node] <= x[k, curr_node])
        @constraint(model, node_U, x[k, curr_node] <= U[k, curr_node])
    end

    if obj_function == 1 && k <= K - 1 # Min, hidden layer
        @objective(model, Min, x[k, curr_node] - s[k, curr_node])
    elseif obj_function == 2 && k <= K - 1 # Max, hidden layer
        @objective(model, Max, x[k, curr_node] - s[k, curr_node])
    elseif obj_function == 1 && k == K # Min, last layer
        @objective(model, Min, x[k, curr_node])
    elseif obj_function == 2 && k == K # Max, last layer
        @objective(model, Max, x[k, curr_node])
    end

    end

    # println("TIME SPENT CREATING MODEL: $(round(creation_time, digits=2)) seconds")

    solve_time = @elapsed optimize!(model)
    solve_time = round(solve_time; sigdigits = 3)

   optimal = nothing
    try
        optimal = objective_value(model)
    catch
        optimal = obj_function == 1 ? current_lower_bounds[curr_node_index] : current_upper_bounds[curr_node_index]
    end

    print(".")

    # set to nothing to speed up garbage collection
    model = nothing

    return optimal, obj_function, curr_node_index
end

"""
bound_tightening_workers(DNN::Chain, init_U_bounds::Vector{Float32}, init_L_bounds::Vector{Float32}, verbose::Bool=false)

A multi-threaded (using workers) implementation of optimal tightened constraint bounds L and U for for a trained DNN.
Using these bounds with the create_JuMP_model function reduces solution time for optimization problems.

# Arguments
- `DNN::Chain`: A trained ReLU DNN.
- `init_U_bounds::Vector{Float32}`: Initial upper bounds on the node values of the DNN.
- `init_L_bounds::Vector{Float32}`: Initial lower bounds on the node values of the DNN.
- `verbose::Bool=false`: Controls Gurobi logs.

# Examples
```julia
L_bounds_workers, U_bounds_workers = bound_tightening_workers(DNN, init_U_bounds, init_L_bounds, false)
```
"""

function bound_tightening_workers(DNN::Chain, init_U_bounds::Vector{Float32}, init_L_bounds::Vector{Float32}, verbose::Bool=false)

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

    for k in 1:K

        # Distributed.pmap returns the bounds in order
        L_U_bounds = Distributed.pmap(node -> bt_workers_inner(K, k, node, W, b, node_count, curr_U_bounds, curr_L_bounds, verbose), 1:(2*node_count[k+1]))

        for node in 1:node_count[k+1]
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

            # L-bounds in 1:node_count[k+1], U-bounds in 1:(node + node_count[k+1])
            curr_L_bounds[curr_node_index] = L_U_bounds[node]
            curr_U_bounds[curr_node_index] = L_U_bounds[node + node_count[k+1]]
        end

    end

    println("Solving optimal constraint bounds using workers complete")

    return curr_U_bounds, curr_L_bounds
end

# Inner function to bound_tightening_workers: assigns a JuMP model to the current worker

function bt_workers_inner(
    K::Int64,
    k::Int64,
    node::Int64,
    W::Vector{Matrix{Float32}},
    b::Vector{Vector{Float32}},
    node_count::Vector{Int64},
    curr_U_bounds::Vector{Float32},
    curr_L_bounds::Vector{Float32},
    verbose::Bool
    )

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
            fix(U[k, j], curr_U_bounds[index], force=true)
            fix(L[k, j], curr_L_bounds[index], force=true)
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
    @assert termination_status(model) == OPTIMAL || termination_status(model) == TIME_LIMIT
        "Problem (layer $k (from 1:$K), node $curr_node, $(obj_function == 1 ? "L" : "U")-bound) is infeasible."
    optimal = objective_value(model)
    println("Worker: $(myid()), layer $k, node $curr_node, $(obj_function == 1 ? "L" : "U")-bound: solve time $(solve_time)s, optimal value $(optimal)")

    return optimal
end

"""
bound_tightening_2workers(DNN::Chain, init_U_bounds::Vector{Float32}, init_L_bounds::Vector{Float32}, verbose::Bool=false)

A multi-threaded (using workers) implementation of optimal tightened constraint bounds L and U for for a trained DNN.
This function uses two in-place models at each layer to reduce memory usage. A max of 2 workers in use simultaneously.
Using these bounds with the create_JuMP_model function reduces solution time for optimization problems.

# Arguments
- `DNN::Chain`: A trained ReLU DNN.
- `init_U_bounds::Vector{Float32}`: Initial upper bounds on the node values of the DNN.
- `init_L_bounds::Vector{Float32}`: Initial lower bounds on the node values of the DNN.
- `verbose::Bool=false`: Controls Gurobi logs.

# Examples
```julia
L_bounds_workers, U_bounds_workers = bound_tightening_2workers(DNN, init_U_bounds, init_L_bounds, false)
```
"""

function bound_tightening_2workers(DNN::Chain, init_U_bounds::Vector{Float32}, init_L_bounds::Vector{Float32}, verbose::Bool=false)

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

    # split the available threads into 2 to be assigned to each worker (integer division)
    n = Threads.nthreads()
    threads_split = [n÷2, n-(n÷2)]

    for k in 1:K

        L_U_bounds = Distributed.pmap(obj_function ->
            bt_2workers_inner(K, k, obj_function, W, b, node_count, curr_U_bounds, curr_L_bounds, threads_split[obj_function], verbose), 1:2)

        curr_L_bounds = L_U_bounds[1]
        curr_U_bounds = L_U_bounds[2]

    end

    println("Solving optimal constraint bounds complete")

    return curr_U_bounds, curr_L_bounds
end


# Inner function to solve_optimal_bounds_2workers: solves L or U bounds for all nodes in a layer using the same JuMP model

function bt_2workers_inner(
    K::Int64,
    k::Int64,
    obj_function::Int64,
    W::Vector{Matrix{Float32}},
    b::Vector{Vector{Float32}},
    node_count::Vector{Int64},
    curr_U_bounds::Vector{Float32},
    curr_L_bounds::Vector{Float32},
    n_threads::Int64,
    verbose::Bool
    )

    curr_U_bounds_copy = copy(curr_U_bounds)
    curr_L_bounds_copy = copy(curr_L_bounds)

    model = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => (verbose ? 1 : 0), "Threads" => n_threads))

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
            fix(U[k, j], curr_U_bounds[index], force=true)
            fix(L[k, j], curr_L_bounds[index], force=true)
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

    for node in 1:node_count[k+1]

        prev_layers_node_sum = 0
        for prev_layer in 0:k-1
            prev_layers_node_sum += node_count[prev_layer+1]
        end
        curr_node_index = prev_layers_node_sum + node

        ### below constraints depending on the node
        temp_sum = sum(W[k][node, j] * x[k-1, j] for j in 1:node_count[k]) # NOTE! prev layer [k]
        if k <= K - 1
            @constraint(model, node_con, temp_sum + b[k][node] == x[k, node] - s[k, node])
            @constraint(model, node_U, x[k, node] <= U[k, node] * z[k, node])
            @constraint(model, node_L, s[k, node] <= -L[k, node] * (1 - z[k, node]))
        elseif k == K # == last value of k
            @constraint(model, node_con, temp_sum + b[k][node] == x[k, node])
            @constraint(model, node_L, L[k, node] <= x[k, node])
            @constraint(model, node_U, x[k, node] <= U[k, node])
        end

        if obj_function == 1 && k <= K - 1 # Min, hidden layer
            @objective(model, Min, x[k, node] - s[k, node])
        elseif obj_function == 2 && k <= K - 1 # Max, hidden layer
            @objective(model, Max, x[k, node] - s[k, node])
        elseif obj_function == 1 && k == K # Min, last layer
            @objective(model, Min, x[k, node])
        elseif obj_function == 2 && k == K # Max, last layer
            @objective(model, Max, x[k, node])
        end

        solve_time = @elapsed optimize!(model)
        solve_time = round(solve_time; sigdigits = 3)
        @assert termination_status(model) == OPTIMAL || termination_status(model) == TIME_LIMIT
            "Problem (layer $k (from 1:$K), node $node, $(obj_function == 1 ? "L" : "U")-bound) is infeasible."
        optimal = objective_value(model)
        println("Worker: $(myid()), layer $k, node $node, $(obj_function == 1 ? "L" : "U")-bound: solve time $(solve_time)s, optimal value $(optimal)")

        # fix the model variable L or U corresponding to the current node to be the optimal value
        if obj_function == 1 # Min
            curr_L_bounds_copy[curr_node_index] = optimal
        elseif obj_function == 2 # Max
            curr_U_bounds_copy[curr_node_index] = optimal
        end

        # deleting and unregistering the constraints assigned to the current node
        delete(model, node_con)
        delete(model, node_L)
        delete(model, node_U)
        unregister(model, :node_con)
        unregister(model, :node_L)
        unregister(model, :node_U)
    end

    if obj_function == 1 # Min
        return curr_L_bounds_copy
    elseif obj_function == 2 # Max
        return curr_U_bounds_copy
    end

end
