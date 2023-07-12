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
    Conv((2,2), 3 => 2, identity),
)

# Conv((a,b), c => d, relu) gives parameters[1] in form a×b×c×d matrix
p = params(DNN)
p[1]
p[3]
# Array order a×b×c×d: a×b image shape, c color channels (RGB 3, grayscale 1, etc.), d image count
# 3×3×1×1 Array{Float32, 4}
# data = Float32[0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9;;;;]
data = Float32[1 0 0; 0 1 0; 0 0 1;;;;]
# data = rand32(3, 3, 1, 1)
DNN(data)

input_size = (3,3)

# function create_CNN_model(DNN::Chain, input_size::Tuple{Int64, Int64}, verbose::Bool=false)

K = length(DNN) # NOTE! there are K+1 layers in the nn
layers = DNN.layers

# store the DNN weights (filters for Conv layers) and biases
DNN_params = Flux.params(DNN)
W = [DNN_params[2*i-1] for i in 1:K]
b = [DNN_params[2*i] for i in 1:K]

function next_sub_img_size(img::Tuple{Int64, Int64}, filter::Tuple{Int64, Int64})
    new_height = img[1] - filter[1] + 1
    new_width = img[2] - filter[2] + 1
    return (new_height, new_width)
end

# store the filter shapes in each layer 1:K
filter_sizes = [size(W[k][:,:,1,1]) for k in 1:K]

# tuples of layer shapes 
sub_img_sizes = Array{Tuple{Int64, Int64}}(undef, K+1)
for k in 1:K+1
    if k == 1
        sub_img_sizes[k] = input_size 
    else 
        sub_img_sizes[k] = next_sub_img_size(sub_img_sizes[k-1], filter_sizes[k-1]) 
    end
end

# stores tuples (img index, img h, img w), such that each convoluted subimage pixel can be accesses
DNN_nodes = Array{Tuple{Int64, Int64, Int64}}(undef, K+1)
for k in 1:K+1
    if k == 1
        DNN_nodes[k] = (1, sub_img_sizes[k]...)
    else 
        DNN_nodes[k] = (size(W[k-1])[4], next_sub_img_size(sub_img_sizes[k-1], (size(W[k-1])[1], size(W[k-1])[2]))...)
    end
end

# model = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => (verbose ? 1 : 0)))
model = Model(optimizer_with_attributes(Gurobi.Optimizer))

# variables x correspond to convolutional layer pixel values: x[k, i, h, w] -> layer, sub img index, img row, img col
@variable(model, x[k in 0:K, i in 1:DNN_nodes[k+1][1], h in 1:DNN_nodes[k+1][2], w in 1:DNN_nodes[k+1][3]] >= 0)
# variables L and U: lower and upper bounds for pixel values (= hidden node values) in the CNN
@variable(model, L[k in 0:K, i in 1:DNN_nodes[k+1][1], h in 1:DNN_nodes[k+1][2], w in 1:DNN_nodes[k+1][3]] == -1000)
@variable(model, U[k in 0:K, i in 1:DNN_nodes[k+1][1], h in 1:DNN_nodes[k+1][2], w in 1:DNN_nodes[k+1][3]] == 1000)


# fix L and U bounds to input nodes
for i in 1:DNN_nodes[1][1]
    for h in 1:DNN_nodes[1][2]
        for w in 1:DNN_nodes[1][3]
            delete_lower_bound(x[0, i, h, w])
            @constraint(model, L[0, i, h, w] <= x[0, i, h, w])
            @constraint(model, x[0, i, h, w] <= U[0, i, h, w])
        end
    end
end

# loop through layers
for k in 1:K
    curr_sub_img_size = sub_img_sizes[k+1] # index k+1 becasue sub_img_sizes contains input size
    curr_filter_size = filter_sizes[k]
    W_rev = reverse(W[k], dims=(1, 2)) # curr layer weights (filters) (rows and columns inverted)

    # loop through number of filters for this (sub)image
    for filter in 1:DNN_nodes[k+1][1]

        # loop through each (sub)image index (i,j) where we place the filter ((1,1) is top left pixel)
        for h in 1:curr_sub_img_size[1]

            # loop through image columns
            for w in 1:curr_sub_img_size[2]
                
                # loop through each (sub)image in the layer
                for i in 1:DNN_nodes[k][1]

                    # here equation for the variable x[k,i,h,w]
                    # println("$k, $i, $h, $w")
                    W_vec = vec(W_rev[:,:,i,filter])

                    x_vec = vec([x[k-1,i,ii,jj] for ii in h:(h+curr_sub_img_size[1]-1), jj in w:(w+curr_sub_img_size[w]-1)])
                    # println(size(W_vec))
                    # println(size(x_vec))
                    mult = W_vec .* x_vec
                    println("x[$k,$filter,$h,$w]: $mult")
                    # println(x_vec)
                    
                    # calculate the value for node x[k,i,h,w]
                    # temp_sum = sum(W_vec[ff] * x[k-1,i,ii,jj] 
                    #     for ii in h:(h+curr_sub_img_size[1]-1), jj in w:(w+curr_sub_img_size[w]-1), ff in 1:sum(curr_filter_size)
                    #         ) + b[1][filter]

                    # println("$k, $i, $h, $w: temp_sum: $temp_sum")
                end
            end
        end
    end
    
end

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

joo = Float32[0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9;;;;]

ei = Float32[-1 -2; -3 -4;;;;]

ei_rev = reverse(ei, dims=(1, 2))

ehkä = Float32[-100, -200]

row, column = size(ei)[1], size(ei)[2]

joojoo = Array{Float32}(undef, row * column)

ind = 1
i, j = 2, 1
for rows in i:row
    for cols in j:column
        joojoo[ind] = joo[:,:,1,1][rows, cols]
        ind += 1
    end
end

eiei = vec(ei_rev)

temp_sum = transpose(joojoo) * eiei + ehkä[1]