#### CONVOLUTIONAL DNNS INTO MILP ###

using Flux, JuMP, Gurobi
using Flux: params
using Random
# Chain: :layers
# Dense: :weight, :bias, :σ
# Conv: :σ, :weight, :bias, :stride, :pad, :dilation, :groups
# MaxPool: :k, :pad, :stride
# MeanPool: :k, :pad, :stride
Random.seed!(42)
DNN = Chain(
    Conv((2,2), 1 => 3, identity),
    Conv((1,1), 3 => 2, identity),
)

# Conv((a,b), c => d, relu) gives parameters[1] in form a×b×c×d matrix
p = params(DNN)
p[1]

# Array order a×b×c×d: a×b image shape, c color channels (RGB 3, grayscale 1, etc.), d image count
# 3×3×1×1 Array{Float32, 4}
# data = Float32[0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9;;;;]
data = Float32[1 0 0; 0 0 0; 0 0 0;;;;]
# data = rand32(3, 3, 1, 1)

input_size = (3,3)

# function create_CNN_model(DNN::Chain, input_size::Tuple{Int64, Int64}, verbose::Bool=false)

    K = length(DNN) # NOTE! there are K+1 layers in the nn
    layers = DNN.layers

    # store the DNN weights and biases
    DNN_params = Flux.params(DNN)
    W = [DNN_params[2*i-1] for i in 1:K]
    b = [DNN_params[2*i] for i in 1:K]

    function next_layer_nodes(img::Tuple{Int64, Int64}, filter::Tuple{Int64, Int64})
        new_height = img[1] - filter[1] + 1
        new_width = img[2] - filter[2] + 1
        return (new_height, new_width)
    end

    # tuples of layer shapes 
    node_count = Array{Tuple{Int64, Int64}}(undef, K+1)
    for k in 0:K
        if k == 0 
            node_count[k+1] = input_size 
        else 
            node_count[k+1] = next_layer_nodes(node_count[k], size(W[k][:,:,1,1])) 
        end
    end

    # stores tuples (img h, img w, number of output images from the current layer)
    DNN_nodes = Array{Tuple{Int64, Int64, Int64}}(undef, K+1)
    for k in 0:K
        if k == 0 
            DNN_nodes[k+1] = (input_size..., 1)
        else 
            DNN_nodes[k+1] = (next_layer_nodes(node_count[k], (size(W[k])[1], size(W[k])[2]))..., size(W[k])[4]*DNN_nodes[k][3])
        end
    end

    # model = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => (verbose ? 1 : 0)))
    model = Model(optimizer_with_attributes(Gurobi.Optimizer))

    # sets the variables x[k,j] and s[k,j], the binary variables z[k,j] and the big-M values U[k,j] and L[k,j]
    @variable(model, x[k in 0:K, h in 1:DNN_nodes[k+1][1], w in 1:DNN_nodes[k+1][2], o in 1:DNN_nodes[k+1][3]] >= 0)

    # @variable(model, x[k in 0:K, j in 1:node_count[k+1]] >= 0)
    # @variable(model, s[k in 1:K, j in 1:node_count[k+1]] >= 0)
    # # @variable(model, z[k in 1:K, j in 1:node_count[k+1]], Bin)
    # @variable(model, U[k in 0:K, j in 1:node_count[k+1]])
    # @variable(model, L[k in 0:K, j in 1:node_count[k+1]]) 

    # # arbitrary lower and upper bounds for all nodes
    # index = 1
    # for k in 0:K
    #     for j in 1:node_count[k+1]
    #         fix(U[k, j], 1000)
    #         fix(L[k, j], -1000)
    #         index += 1
    #     end
    # end

    # # fix bounds U and L to input nodes
    # for input_node in 1:node_count[1]
    #     delete_lower_bound(x[0, input_node])
    #     @constraint(model, L[0, input_node] <= x[0, input_node])
    #     @constraint(model, x[0, input_node] <= U[0, input_node])
    # end

    # # constraints corresponding to the ReLU activation functions
    # for k in 1:K
    #     for node in 1:node_count[k+1] # node count of the next layer of k, i.e., the layer k+1
    #         temp_sum = sum(W[k][node, j] * x[k-1, j] for j in 1:node_count[k])
    #         if k < K # hidden layers: k = 1, ..., K-1
    #             @constraint(model, temp_sum + b[k][node] == x[k, node] - s[k, node])
    #         else # output layer: k == K
    #             @constraint(model, temp_sum + b[k][node] == x[k, node])
    #         end
    #     end
    # end

    # # fix bounds to the hidden layer nodes
    # @constraint(model, [k in 1:K, j in 1:node_count[k+1]], x[k, j] <= U[k, j] * z[k, j])
    # @constraint(model, [k in 1:K, j in 1:node_count[k+1]], s[k, j] <= -L[k, j] * (1 - z[k, j]))

    # # fix bounds to the output nodes
    # for output_node in 1:node_count[K+1]
    #     delete_lower_bound(x[K, output_node])
    #     @constraint(model, L[K, output_node] <= x[K, output_node])
    #     @constraint(model, x[K, output_node] <= U[K, output_node])
    # end

    @objective(model, Max, x[K, 1]) # arbitrary objective function to have a complete JuMP model

    # return model
# end

# reversed_matrix = reverse(matrix, dims=(1, 2))