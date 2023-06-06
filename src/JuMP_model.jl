"""
create_JuMP_model(DNN::Chain, bounds_U::Vector{Float32}, bounds_L::Vector{Float32}, bound_tightening::Bool=false)

Converts a ReLU DNN to a 0-1 MILP formulatuion

# Arguments
- `DNN::Chain`: A trained ReLU DNN.
- `U_bounds::Vector{Float32}`: Upper bounds on the node values of the DNN.
- `L_bounds::Vector{Float32}`: Lower bounds on the node values of the DNN.
- `bound_tightening::Bool=false`: Optional bound tightening of the constraint bounds

# Examples
```julia
model = create_JuMP_model(DNN, U_bounds, L_bounds, true)
```
"""
function create_JuMP_model(DNN::Chain, U_bounds::Vector{Float32}, L_bounds::Vector{Float32}, bound_tightening::Bool=false)

    K = length(DNN) # NOTE! there are K+1 layers in the nn
    for i in 1:K-1
        @assert DNN[i].σ == relu "Hidden layers must use 'relu' as the activation function"
    end
    @assert DNN[K].σ == identity "Output layer must use the 'identity' activation function"

    # store the DNN weights and biases
    DNN_params = params(DNN)
    W = [DNN_params[2*i-1] for i in 1:K]
    b = [DNN_params[2*i] for i in 1:K]

    # stores the node count of layer k (starting at layer k=0) at index k+1
    input_node_count = length(DNN_params[1][1, :])
    node_count = [if k == 1 input_node_count else length(DNN_params[2*(k-1)]) end for k in 1:K+1]

    final_L_bounds = copy(L_bounds)
    final_U_bounds = copy(U_bounds)

    # optional: calculates optimal lower and upper bounds L and U 
    if bound_tightening 
        final_L_bounds, final_U_bounds = solve_optimal_bounds(DNN, U_bounds, L_bounds)
    end

    model = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 1))

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
            fix(U[k, j], final_U_bounds[index])
            fix(L[k, j], final_L_bounds[index])
            index += 1
        end
    end

    # fix bounds to the input nodes
    for input_node in 1:node_count[1]
        delete_lower_bound(x[0, input_node])
        @constraint(model, L[0, input_node] <= x[0, input_node])
        @constraint(model, x[0, input_node] <= U[0, input_node])
    end

    # constraints corresponding to the activation functions
    for k in 1:K
        for node in 1:node_count[k+1] # node count of the next layer of k, i.e., the layer k+1
            temp_sum = sum(W[k][node, j] * x[k-1, j] for j in 1:node_count[k])
            if k < K # constraint (4b) (k=1, ..., k=K-1)
                @constraint(model, temp_sum + b[k][node] == x[k, node] - s[k, node])
            elseif k == K # constraint (4e) (k=K)
                @constraint(model, temp_sum + b[k][node] == x[k, node])
            end
        end
    end

    # fix bounds to the hidden layer nodes
    @constraint(model, [k in 1:K, j in 1:node_count[k+1]], x[k, j] <= U[k, j] * z[k, j])
    @constraint(model, [k in 1:K, j in 1:node_count[k+1]], s[k, j] <= -L[k, j] * (1 - z[k, j]))

    # fix bounds to the output nodes
    for output_node in 1:node_count[K+1]
        delete_lower_bound(x[K, output_node])
        @constraint(model, L[K, output_node] <= x[K, output_node])
        @constraint(model, x[K, output_node] <= U[K, output_node])
    end

    @objective(model, Max, x[K, 1]) # arbitrary objective function to have a complete JuMP model

    return model
end

"""
evaluate!(JuMP_model::Model, input::Vector{Float32})

Fixes the variables corresponding to the DNN input to a given input vector.

# Arguments
- `JuMP_model::Model`: A JuMP model representing a traied ReLU DNN (generated using the function create_JuMP_model).
- `input::Vector{Float32}`: A given input to the trained DNN.

# Examples
```julia
evaluate!(JuMP_model, input)
```
"""
# fixes the input values (layer k=0) for the JuMP model
function evaluate!(JuMP_model::Model, input::Vector{Float32})
    x = JuMP_model[:x] # stores the @variable with name x from the JuMP model
    input_len = length(input)
    @assert input_len == length(x[0,:]) "'input' has wrong dimension"
    for input_node in 1:input_len
        fix(x[0, input_node], input[input_node], force=true) # fix value of input to x[0,j]
    end
end