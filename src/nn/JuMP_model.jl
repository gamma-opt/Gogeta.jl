using JuMP, Flux, Gurobi
using Flux: params

"""
create_JuMP_model(DNN::Chain, L_bounds::Vector{Float32}, U_bounds::Vector{Float32}, bound_tightening::String="none", bt_verbose::Bool=false)

Converts a ReLU DNN to a 0-1 MILP JuMP model. The ReLU DNN is assumed to be a Flux.Chain.
The activation function must be "relu" in all hidden layers and "identity" in the output layer.
The lower and upper bounds to the function are given as a Vector{Float32}. The bounds are in order from the input layer to the output layer.
The keyword argument "bt" determines if bound tightening is to be used on the constraint bounds:
— "none": No bound tightening is used.
— "singletread": One shared JuMP model is used to calculate bounds one at a time.
— "threads": The bounds are calculated using a separate model for each subproblem using Threads
— "workers": The bounds are calculated using a separate model for each subproblem using Workers
— "2 workers": The bounds are calculated with a maximum of two workers at each layer (upper and lower bounds). Each worker reuses the JuMP model.

# Arguments
- `DNN::Chain`: A trained ReLU DNN.
- `L_bounds::Vector{Float32}`: Lower bounds on the node values of the DNN.
- `U_bounds::Vector{Float32}`: Upper bounds on the node values of the DNN.
- `bt::String="none"`: Optional bound tightening of the constraint bounds. Can be set to "none", "singlethread", "threads", "workers" or "2 workers".
- `bt_verbose::Bool=false`: Controls Gurobi logs in bound tightening procedures.

# Examples
```julia
model = create_JuMP_model(DNN, L_bounds, U_bounds, "singlethread", false)
```
"""
function create_JuMP_model(DNN::Chain, L_bounds::Vector{Float32}, U_bounds::Vector{Float32}, bt::String="none", bt_verbose::Bool=false)

    K = length(DNN) # NOTE! there are K+1 layers in the nn
    for i in 1:K-1
        @assert DNN[i].σ == relu "Hidden layers must use \"relu\" as the activation function"
    end
    @assert DNN[K].σ == identity "Output layer must use the \"identity\" activation function"

    # store the DNN weights and biases
    DNN_params = Flux.params(DNN)
    W = [DNN_params[2*i-1] for i in 1:K]
    b = [DNN_params[2*i] for i in 1:K]

    # stores the node count of layer k (starting at layer k=0) at index k+1
    input_node_count = length(DNN_params[1][1, :])
    node_count = [if k == 1 input_node_count else length(DNN_params[2*(k-1)]) end for k in 1:K+1]

    final_L_bounds = copy(L_bounds)
    final_U_bounds = copy(U_bounds)

    # optional: calculates optimal lower and upper bounds L and U
    @assert bt == "none" || bt == "singlethread" || bt == "threads" || bt == "workers" || bt == "2 workers"
        "bound_tightening has to be set to \"none\", \"singlethread\", \"threads\", \"workers\" or \"2 workers\"."
    if bt == "singlethread"
        final_L_bounds, final_U_bounds = bound_tightening(DNN, U_bounds, L_bounds, bt_verbose)
    elseif bt == "threads"
        final_L_bounds, final_U_bounds = bound_tightening_threads(DNN, U_bounds, L_bounds, bt_verbose)
    elseif bt == "workers"
        final_L_bounds, final_U_bounds = bound_tightening_workers(DNN, U_bounds, L_bounds, bt_verbose)
    elseif bt == "2 workers"
        final_L_bounds, final_U_bounds = bound_tightening_2workers(DNN, U_bounds, L_bounds, bt_verbose)
    end

    model = Model(optimizer_with_attributes(Gurobi.Optimizer))

    # sets the variables x[k,j] and s[k,j], the binary variables z[k,j] and the big-M values U[k,j] and L[k,j]
    @variable(model, x[k in 0:K, j in 1:node_count[k+1]] >= 0)
    if K > 1 # s and z variables only to hidden layers, i.e., layers 1:K-1
        @variable(model, s[k in 1:K-1, j in 1:node_count[k+1]] >= 0)
        @variable(model, z[k in 1:K-1, j in 1:node_count[k+1]], Bin)
    end
    @variable(model, U[k in 0:K, j in 1:node_count[k+1]])
    @variable(model, L[k in 0:K, j in 1:node_count[k+1]])

    # fix values to all U[k,j] and L[k,j] from U_bounds and L_bounds
    index = 1
    for k in 0:K
        for j in 1:node_count[k+1]
            fix(U[k, j], final_U_bounds[index])
            fix(L[k, j], final_L_bounds[index])
            index += 1
        end
    end

    # fix bounds U and L to input nodes
    for input_node in 1:node_count[1]
        delete_lower_bound(x[0, input_node])
        @constraint(model, L[0, input_node] <= x[0, input_node])
        @constraint(model, x[0, input_node] <= U[0, input_node])
    end

    # constraints corresponding to the ReLU activation functions
    for k in 1:K
        for node in 1:node_count[k+1] # node count of the next layer of k, i.e., the layer k+1
            temp_sum = sum(W[k][node, j] * x[k-1, j] for j in 1:node_count[k])
            if k < K # hidden layers: k = 1, ..., K-1
                @constraint(model, temp_sum + b[k][node] == x[k, node] - s[k, node])
            else # output layer: k == K
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

    @objective(model, Max, x[1, 1]) # arbitrary objective function to have a complete JuMP model

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

function evaluate!(JuMP_model::Model, input::Vector{Float32})
    x = JuMP_model[:x] # stores the @variable with name x from the JuMP_model
    input_len = length(input)
    @assert input_len == length(x[0, :]) "\"input\" has wrong dimension"
    for input_node in 1:input_len
        fix(x[0, input_node], input[input_node], force=true) # fix value of input to x[0,j]
    end
end