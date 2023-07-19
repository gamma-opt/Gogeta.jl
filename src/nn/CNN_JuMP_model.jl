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
# DNN = Chain(
#     Conv((2,2), 1 => 2, relu, bias = rand32(2)),
#     MeanPool((2,2)),
#     Flux.flatten,
#     Dense(8 => 9, relu),
#     Dense(9 => 10),
# )
DNN = Chain(
    Conv((5,5), 3=>16, relu),
    MeanPool((2,2)),
    Conv((5,5), 16=>8, relu),
    MeanPool((2,2)),
    Flux.flatten,
    Dense(200 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
)

# Conv((a,b), c => d, relu) gives parameters[1] in form a×b×c×d matrix
p = params(DNN)
p[1]
# p[3]
# Array order a×b×c×d: a×b image shape, c color channels (RGB 3, grayscale 1, etc.), d image count
# 3×3×1×1 Array{Float32, 4}
data = Float32[0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9;;; 0.11 0.22 0.33; 0.44 0.55 0.66; 0.77 0.88 0.99;;;;]
data = Float32[0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9;;;;]
data = Float32[0.1 0.2 0.3 0.4; 0.4 0.5 0.6 0.6; 0.7 0.8 0.9 0.9; 0.7 0.8 0.9 0.9;;;;]

data = Float32[0.1 0.2; 0.3 0.4;;; 0.5 0.6; 0.7 0.8;;;;]
# data = Float32[1 0 0; 0 1 0; 0 0 1;;;;]
data = rand32(5, 5, 1, 1)
data = rand32(10, 10, 3, 1)
data = rand32(32, 32, 3, 1)

DNN(data)



input_size = (size(data)[3], size(data)[1], size(data)[2])

# function create_CNN_model(DNN::Chain, input_size::Tuple{Int64, Int64}, verbose::Bool=false)
layers = DNN.layers
layers_no_flatten = Tuple(filter(x -> typeof(x) != typeof(Flux.flatten), layers))
K = 0 # number of layers before Flux.flatten (K+1 layers before flatten)
D = 0 # number of dense layers after Flux.flatten
for layer in layers_no_flatten
    if isa(layer, Conv) || isa(layer, MaxPool) || isa(layer, MeanPool)
        K += 1
    elseif isa(layer, Dense)
        D += 1
    end
end

# store the DNN weights (filters for Conv layers) and biases
# for non Conv or Dense layers, stores [NaN32] so that weight and bias indexing stays consistent
DNN_params = Flux.params(DNN)
W = Vector{Array{Float32}}(undef, K+D)
b = Vector{Vector{Float32}}(undef, K+D)
idx = 1
for k in 1:K+D
    if isa(layers_no_flatten[k], Conv) || isa(layers_no_flatten[k], Dense)
        W[k] = DNN_params[2*idx-1]
        b[k] = DNN_params[2*idx]
        idx += 1
    else 
        # poolingl layers do not have weights or biases associated with them, these are just
        # filler values that are never used in the JuMP model
        W[k] = [NaN32]
        b[k] = [NaN32]
    end
end

# store the filter shapes in each layer 1:K (Conv and MeanPool)
filter_sizes = Vector{Tuple{Int64, Int64}}(undef, K)
for k in 1:K
    if isa(layers[k], Conv)
        filter_sizes[k] = size(W[k][:,:,1,1])
    elseif isa(layers[k], MeanPool) # || isa(layers[k], MaxPool)
        filter_sizes[k] = layers[k].k # .k = pooling layer shape
    end
end

# new img size after passing through 1) Conv layer filter or 2) a MeanPool layer
function next_sub_img_size(img::Tuple{Int64, Int64}, filter::Tuple{Int64, Int64}, pooling_layer::Bool = false)
    new_height = pooling_layer ? div(img[1], filter[1]) : (img[1] - filter[1] + 1)
    new_width  = pooling_layer ? div(img[2], filter[2]) : (img[2] - filter[2] + 1)
    return (new_height, new_width)
end

# For Conv and MeanPool layers: stores tuples (img index, img h, img w), such that each convoluted subimage pixel can be accesses
# For Dense layers: stores the number of nodes in a layer at index 1 (number of nodes, 1, 1)
DNN_nodes = Array{Tuple{Int64, Int64, Int64}}(undef, K+1+D)
for k in 1:K+1+D
    if k == 1
        DNN_nodes[k] = input_size
    else
        if isa(layers[k-1], Conv)
            DNN_nodes[k] = (size(W[k-1])[4], next_sub_img_size(DNN_nodes[k-1][2:3], filter_sizes[k-1])...)
        elseif isa(layers[k-1], MaxPool) || isa(layers[k-1], MeanPool)
            DNN_nodes[k] = (DNN_nodes[k-1][1], next_sub_img_size(DNN_nodes[k-1][2:3], filter_sizes[k-1], true)...)
        elseif k > K+1 # Dense layer
            DNN_nodes[k] = (size(layers[k].weight)[1], 1, 1)
        end
    end
end

# stored the sub image sizes (h,w) at each Conv and MeanPool layer (there are often multiple sub images at each Conv layer)
sub_img_sizes = Array{Tuple{Int64, Int64}}(undef, K+1)
for k in 1:K+1
    sub_img_sizes[k] = DNN_nodes[k][2:3]
end

# model = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => (verbose ? 1 : 0)))
model = Model(optimizer_with_attributes(Gurobi.Optimizer))

# variables x correspond to convolutional layer pixel values: x[k, i, h, w] -> layer, sub img index, img row, img col
@variable(model, x[k in 0:K+D, i in 1:DNN_nodes[k+1][1], h in 1:DNN_nodes[k+1][2], w in 1:DNN_nodes[k+1][3]] >= 0)
if K+D > 1 # s and z variables only to hidden layers, i.e., when K+D > 1
    @variable(model, s[k in 1:K-1+D, i in 1:DNN_nodes[k+1][1], h in 1:DNN_nodes[k+1][2], w in 1:DNN_nodes[k+1][3]] >= 0)
    @variable(model, z[k in 1:K-1+D, i in 1:DNN_nodes[k+1][1], h in 1:DNN_nodes[k+1][2], w in 1:DNN_nodes[k+1][3]], Bin)
end
# variables L and U: lower and upper bounds for pixel values (= hidden node values) in the CNN
@variable(model, L[k in 0:K+D, i in 1:DNN_nodes[k+1][1], h in 1:DNN_nodes[k+1][2], w in 1:DNN_nodes[k+1][3]] == -1000)
@variable(model, U[k in 0:K+D, i in 1:DNN_nodes[k+1][1], h in 1:DNN_nodes[k+1][2], w in 1:DNN_nodes[k+1][3]] == 1000)


# delete lower bound and fix L and U bounds to input nodes
for i in 1:DNN_nodes[1][1]
    for h in 1:DNN_nodes[1][2]
        for w in 1:DNN_nodes[1][3]
            delete_lower_bound(x[0, i, h, w])
            @constraint(model, L[0, i, h, w] <= x[0, i, h, w])
            @constraint(model, x[0, i, h, w] <= U[0, i, h, w])
        end
    end
end

# delete lower bound and fix L and U bounds to output nodes
for i in 1:DNN_nodes[K+1+D][1]
    for h in 1:DNN_nodes[K+1+D][2]
        for w in 1:DNN_nodes[K+1+D][3]
            delete_lower_bound(x[K+D, i, h, w])
            @constraint(model, L[K+D, i, h, w] <= x[K+D, i, h, w])
            @constraint(model, x[K+D, i, h, w] <= U[K+D, i, h, w])
        end
    end
end

# loop through Conv and MeanPool layers
for k in 1:K
    curr_sub_img_size = sub_img_sizes[k+1] # index k+1 becasue sub_img_sizes contains input size
    curr_filter_size = filter_sizes[k]

    if isa(layers[k], Conv)

        W_rev = reverse(W[k], dims=(1, 2)) # curr layer weights (filters) (rows and columns inverted)

        # loop through number of filters for this (sub)image
        for filter in 1:DNN_nodes[k+1][1]

            # loop through each subimage index (h,w) in the following layer
            for h in 1:curr_sub_img_size[1]
                for w in 1:curr_sub_img_size[2]

                    var_expression_count = DNN_nodes[k][1] * reduce(*, curr_filter_size)
                    var_expression = Array{AffExpr}(undef, var_expression_count)
                    index = 1

                    # loop through each (sub)image in the layer
                    for i in 1:DNN_nodes[k][1]

                        # here equation for the variable x[k,i,h,w]

                        W_vec = vec(W_rev[:,:,i,filter])
                        x_vec = vec([x[k-1,i,ii,jj] for ii in h:(h+curr_filter_size[1]-1), jj in w:(w+curr_filter_size[2]-1)])
                        # println("h: $h, curr_filter_size[1]: $(curr_filter_size[1]), w: $w, curr_filter_size[2]: $(curr_filter_size[2])")
                        mult = W_vec .* x_vec

                        for expr in 1:reduce(*, curr_filter_size)
                            var_expression[index] = mult[expr]
                            index += 1
                        end
                    end

                    temp_sum = sum(var_expression)
                    if k < K # hidden layers: k = 1, ..., K-1
                        @constraint(model, temp_sum + b[k][filter] == x[k, filter, h, w] - s[k, filter, h, w])
                    else # output layer: k == K
                        @constraint(model, temp_sum + b[k][filter] == x[k, filter, h, w])
                    end
                end
            end
        end

    elseif isa(layers[k], MeanPool)

        # loop through each subimage index (h,w) in the following layer
        for h in 1:curr_sub_img_size[1]
            for w in 1:curr_sub_img_size[2]

                # loop through each (sub)image in the layer
                for i in 1:DNN_nodes[k][1]

                    # temp_sum - equation for the variable x[k,i,h,w] (mean of the values under the filter)
                    h_filter = curr_filter_size[1]
                    w_filter = curr_filter_size[2]
                    x_vec = vec([x[k-1,i,hh,ww] for hh in (h_filter*(h-1)+1):(h_filter*h), ww in (w_filter*(w-1)+1):(w_filter*w)])

                    temp_sum = sum(x_vec) / reduce(*, curr_filter_size)
                    @constraint(model, temp_sum == x[k, i, h, w])
                end
            end
        end
    end
end

# loop through the Dense layers that are after the Flux.flatten "pseudolayer"
for k in K+1:K+D
    for node in 1:DNN_nodes[k+1][1]
        temp_sum = 0
        if k == K+1 # previous layer nodes still in matrix form, reshaping for to a vector
            x_vec = vec([x[k-1,i,h,w] for h in 1:DNN_nodes[k][2], w in 1:DNN_nodes[k][3], i in 1:DNN_nodes[k][1]])
            temp_sum = sum(W[k][node, j] * x_vec[j] for j in 1:reduce(*, DNN_nodes[k]))
        else
            temp_sum = sum(W[k][node, j] * x[k-1, j, 1, 1] for j in 1:reduce(*, DNN_nodes[k][1]))
        end
        if k < K+D # hidden Dense layer
            @constraint(model, temp_sum + b[k][node] == x[k, node, 1, 1] - s[k, node, 1, 1])
        else # output layer
            @constraint(model, temp_sum + b[k][node] == x[k, node, 1, 1])
        end
    end
end

for k in 1:length(layers_no_flatten)-1
    # fix bounds to the hidden layers (Conv and Dense) with relu activation function
    if isa(layers_no_flatten[k], Conv) || isa(layers_no_flatten[k], Dense)
        # println(k)
        @constraint(model, [[k], i in 1:DNN_nodes[k+1][1], h in 1:DNN_nodes[k+1][2], w in 1:DNN_nodes[k+1][3]], 
                            x[k, i, h, w] <= U[k, i, h, w] * z[k, i, h, w])
        @constraint(model, [[k], i in 1:DNN_nodes[k+1][1], h in 1:DNN_nodes[k+1][2], w in 1:DNN_nodes[k+1][3]], 
                            s[k, i, h, w] <= -L[k, i, h, w] * (1 - z[k, i, h, w]))
    end
end

# fix input values to known data (testing purposes only!)
for i in 1:DNN_nodes[1][1]
    for h in 1:DNN_nodes[1][2]
        for w in 1:DNN_nodes[1][3]
            fix(x[0,i,h,w], data[h,w,i,1], force=true)
        end
    end
end

# arbitrary objective function to allow optimization
@objective(model, Max, x[1,1,1,1])

# end

optimize!(model)

# extract output values from the JuMP model, same as from the CNN
function extract_output(model::Model, DNN_nodes)
    x = model[:x]
    output = []
    len = length(DNN_nodes)
    for i in 1:DNN_nodes[len][1]
        for w in 1:DNN_nodes[len][3]
            for h in 1:DNN_nodes[len][2]
                push!(output, value(x[len-1,i,h,w]))
            end
        end
    end
    return output
end

output = extract_output(model, DNN_nodes)

DNN(data)
