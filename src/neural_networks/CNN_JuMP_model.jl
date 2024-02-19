using Flux, JuMP, Gurobi
using Flux: params

"""
    create_CNN_JuMP_model(CNN::Chain, data_shape::Tuple{Int64, Int64, Int64, Int64}, L_bounds::Vector{Array{Float32}}, U_bounds::Vector{Array{Float32}})

Converts a CNN with ReLU activation functions to a 0-1 MILP JuMP model. The ReLU CNN is assumed to be a Flux.Chain.
The activation function must be "relu" in all hidden (Conv and Dense) layers and "identity" in the output layer.
The model assumes the following things from the underlying CNN:
- Input must be Array{Float32, 4}, (e.g. {32, 32, 3, 1}) where first two indices are height and width of data, 
third index is channel count (e.g. 1 for grayscale image, 3 for RGB), fourth index is batch size (here must be 1)
- The CNN must consist of 3 sections: the convolutional and pooling layers, Flux.flatten, and dense layers.
- Convolutional layers must use relu, pooling layers must be MaxPool or MeanPool. 
- Layers must use default setting, such as stride, pad, dilation, etc. Conv.filter, MaxPool.k and MeanPool.k (window) sizes can be arbitrary.
- No convolutional or pooling layers are necessary before Flux.flatten. Also, no dense layers are necessary after Flux.flatten.

# Arguments
- `CNN::Chain`: A trained ReLU CNN with the above assumptions
- `data_shape::Tuple{Int64, Int64, Int64, Int64}`: Shape of the data used in the CNN as a Tuple, e.g., (32, 32, 3, 1) (similar logis as above)
- `L_bounds::Vector{Array{Float32}}`: Lower bound big-M values for contraint bounds, indexed L_bounds[layer][channel, height, width]
- `U_bounds::Vector{Array{Float32}}`: Uower bound big-M values for contraint bounds, indexed U_bounds[layer][channel, height, width]

# Examples
```julia
model = create_CNN_JuMP_model(CNN, data_shape, L_bounds, U_bounds)
```
"""
function create_CNN_JuMP_model(CNN::Chain, data_shape::Tuple{Int64, Int64, Int64, Int64}, L_bounds::Vector{Array{Float32}}, U_bounds::Vector{Array{Float32}})

    layers = CNN.layers
    layers_no_flatten = Tuple(filter(x -> typeof(x) != typeof(Flux.flatten), layers))
    K = 0 # number of layers before Flux.flatten (K+1 layers before flatten as input is layer 0)
    D = 0 # number of dense layers after Flux.flatten
    n_max = 0 # number of MaxPool layers
    for layer in layers_no_flatten
        if isa(layer, Conv) || isa(layer, MeanPool)
            K += 1
        elseif isa(layer, MaxPool)
            K += 1
            n_max += 1
        elseif isa(layer, Dense)
            D += 1
        end
    end

    # store the CNN weights (= filters for Conv layers) and biases
    # for non Conv or Dense layers, stores [NaN32] so that weight and bias indexing stays consistent
    CNN_params = params(CNN)
    W = Vector{Array{Float32}}(undef, K+D)
    b = Vector{Vector{Float32}}(undef, K+D)
    idx = 1
    for k in 1:K+D
        if isa(layers_no_flatten[k], Conv) || isa(layers_no_flatten[k], Dense)
            W[k] = CNN_params[2*idx-1]
            b[k] = CNN_params[2*idx]
            idx += 1
        else 
            # pooling layers do not have weights or biases associated with them, these are just
            # filler values that are never used in the JuMP model
            W[k] = [NaN32]
            b[k] = [NaN32]
        end
    end

    # store the filter shapes in each layer 1:K (Conv and MeanPool layers)
    filter_sizes = Vector{Tuple{Int64, Int64}}(undef, K)
    for k in 1:K
        if isa(layers_no_flatten[k], Conv)
            filter_sizes[k] = size(W[k][:,:,1,1])
        elseif isa(layers_no_flatten[k], MaxPool) || isa(layers_no_flatten[k], MeanPool)
            filter_sizes[k] = layers_no_flatten[k].k # .k = pooling layer shape
        end
    end

    # For Conv, MaxPool and MeanPool layers: stores tuples (img idx, img h, img w), such that each convoluted subimage pixel can be accesses
    # For Dense layers: stores the number of nodes in a layer at index 1, i.e. tuples (number of nodes, 1, 1)
    CNN_nodes = Array{Tuple{Int64, Int64, Int64}}(undef, K+1+D)
    for k in 1:K+1+D
        if k == 1
            CNN_nodes[k] = (data_shape[3], data_shape[1], data_shape[2])
        else
            if isa(layers_no_flatten[k-1], Conv)
                CNN_nodes[k] = (size(W[k-1])[4], next_sub_img_size(CNN_nodes[k-1][2:3], filter_sizes[k-1])...)
            elseif isa(layers_no_flatten[k-1], MaxPool) || isa(layers_no_flatten[k-1], MeanPool)
                CNN_nodes[k] = (CNN_nodes[k-1][1], next_sub_img_size(CNN_nodes[k-1][2:3], filter_sizes[k-1], true)...)
            elseif k > K+1 # Dense layer
                CNN_nodes[k] = (size(layers_no_flatten[k-1].weight)[1], 1, 1)
            end
        end
    end

    # stores the sub image sizes (h,w) at each Conv, MaxPool and MeanPool layer (although there are often multiple sub images at each layer)
    sub_img_sizes = Array{Tuple{Int64, Int64}}(undef, K+1)
    for k in 1:K+1
        sub_img_sizes[k] = CNN_nodes[k][2:3]
    end

    model = Model(optimizer_with_attributes(Gurobi.Optimizer))

    # variables x correspond to convolutional layer pixel values: x[k,i,h,w] such that the indices are: [layer, sub img idx, img row, img col]
    @variable(model, x[k in 0:K+D, i in 1:CNN_nodes[k+1][1], h in 1:CNN_nodes[k+1][2], w in 1:CNN_nodes[k+1][3]] >= 0)
    if K+D > 1 # s and z variables only to hidden layers, i.e., layers 1:K+D-1
        @variable(model, s[k in 1:K+D-1, i in 1:CNN_nodes[k+1][1], h in 1:CNN_nodes[k+1][2], w in 1:CNN_nodes[k+1][3]] >= 0)
        @variable(model, z[k in 1:K+D-1, i in 1:CNN_nodes[k+1][1], h in 1:CNN_nodes[k+1][2], w in 1:CNN_nodes[k+1][3]], Bin)
    end
    # variables L and U: lower and upper bounds for pixel values (= hidden node values) in the CNN
    @variable(model, L[k in 0:K+D, i in 1:CNN_nodes[k+1][1], h in 1:CNN_nodes[k+1][2], w in 1:CNN_nodes[k+1][3]])
    @variable(model, U[k in 0:K+D, i in 1:CNN_nodes[k+1][1], h in 1:CNN_nodes[k+1][2], w in 1:CNN_nodes[k+1][3]])

    # fix the values of JuMP variables L and U from L_bounds and U_bounds
    for k in 0:K+D
        for i in 1:CNN_nodes[k+1][1]
            for h in 1:CNN_nodes[k+1][2]
                for w in 1:CNN_nodes[k+1][3]
                    fix(L[k,i,h,w], L_bounds[k+1][i,h,w])
                    fix(U[k,i,h,w], U_bounds[k+1][i,h,w])
                end
            end
        end
    end

    if n_max > 0 # additional binary variables are required to implement MaxPool layers
        @variable(model, z_max[k in 0:K-1, i in 1:CNN_nodes[k+1][1], h in 1:CNN_nodes[k+1][2], w in 1:CNN_nodes[k+1][3]], Bin)

        # remove MaxPool binary variables from non-MaxPool layers
        for k in 0:K-1
            if !isa(layers_no_flatten[k+1], MaxPool)
                for i in 1:CNN_nodes[k+1][1]
                    for h in 1:CNN_nodes[k+1][2]
                        for w in 1:CNN_nodes[k+1][3]
                            delete(model, z_max[k,i,h,w])
                        end
                    end
                end
            end
        end
    end

    # remove "ReLU"-variables from MaxPool and MeanPool layers
    for k in 1:K
        if isa(layers_no_flatten[k], MaxPool) || isa(layers_no_flatten[k], MeanPool)
            for i in 1:CNN_nodes[k+1][1]
                for h in 1:CNN_nodes[k+1][2]
                    for w in 1:CNN_nodes[k+1][3]
                        delete(model, s[k,i,h,w])
                        delete(model, z[k,i,h,w])
                        delete(model, L[k,i,h,w])
                        delete(model, U[k,i,h,w])
                    end
                end
            end
        end
    end

    # delete lower bound and fix L and U bounds to input nodes
    for i in 1:CNN_nodes[1][1]
        for h in 1:CNN_nodes[1][2]
            for w in 1:CNN_nodes[1][3]
                delete_lower_bound(x[0,i,h,w])
                @constraint(model, L[0,i,h,w] <= x[0,i,h,w])
                @constraint(model, x[0,i,h,w] <= U[0,i,h,w])
            end
        end
    end

    # delete lower bound and fix L and U bounds to output nodes
    for i in 1:CNN_nodes[K+1+D][1]
        for h in 1:CNN_nodes[K+1+D][2]
            for w in 1:CNN_nodes[K+1+D][3]
                delete_lower_bound(x[K+D,i,h,w])
                @constraint(model, L[K+D,i,h,w] <= x[K+D,i,h,w])
                @constraint(model, x[K+D,i,h,w] <= U[K+D,i,h,w])
            end
        end
    end

    # loop through Conv, MaxPool and MeanPool layers
    for k in 1:K
        curr_sub_img_size = sub_img_sizes[k+1] # index k+1 becasue sub_img_sizes contains input size
        curr_filter_size = filter_sizes[k]

        if isa(layers_no_flatten[k], Conv)
            W_rev = reverse(W[k], dims=(1, 2)) # curr layer weights (i.e., filters) (rows and columns are inverted!)

            # loop through number of filters for this (sub)image
            for filter in 1:CNN_nodes[k+1][1]

                # loop through each subimage index (h,w) in the following layer
                # a filter in this layer is placed at each sub img pixel index from the following layer
                for h in 1:curr_sub_img_size[1]
                    for w in 1:curr_sub_img_size[2]

                        var_expression_count = CNN_nodes[k][1] * reduce(*, curr_filter_size)
                        var_expression = Array{AffExpr}(undef, var_expression_count)
                        idx = 1

                        # loop through each (sub)image in the layer
                        for i in 1:CNN_nodes[k][1]

                            # here equation for the variable x[k,i,h,w], which is gotten by placing the filter at the right index
                            # and calculating the weighted sum of the pixel values
                            W_vec = vec(W_rev[:,:,i,filter])
                            x_vec = vec([x[k-1,i,ii,jj] for ii in h:(h+curr_filter_size[1]-1), jj in w:(w+curr_filter_size[2]-1)])
                            mult = W_vec .* x_vec

                            for expr in 1:reduce(*, curr_filter_size)
                                var_expression[idx] = mult[expr]
                                idx += 1
                            end
                        end

                        temp_sum = sum(var_expression)
                        if k < K 
                            @constraint(model, temp_sum + b[k][filter] == x[k,filter,h,w] - s[k,filter,h,w])
                        else # output layer: k == K
                            @constraint(model, temp_sum + b[k][filter] == x[k,filter,h,w])
                        end
                    end
                end
            end

        else # MaxPool and/or MeanPool layers

            # loop through each subimage index (h,w) in the following layer
            for h in 1:curr_sub_img_size[1]
                for w in 1:curr_sub_img_size[2]

                    # loop through each (sub)image in the layer
                    for i in 1:CNN_nodes[k][1]
                        
                        h_filter = curr_filter_size[1]
                        w_filter = curr_filter_size[2]

                        if isa(layers_no_flatten[k], MaxPool) # pick the maximum of the pixel values over the filter

                            z_vec = Array{VariableRef}(undef, h_filter * w_filter)
                            idx = 1

                            # loop through each index in the previous layer that is under the filter
                            for hh in (h_filter*(h-1)+1):(h_filter*h)
                                for ww in (w_filter*(w-1)+1):(w_filter*w)
                                    @constraint(model, x[k-1,i,hh,ww] <= x[k,i,h,w])

                                    # indicator constraint to choose the maximum value
                                    @constraint(model, z_max[k-1,i,hh,ww] => {x[k-1,i,hh,ww] >= x[k,i,h,w]})
                                    z_vec[idx] = z_max[k-1,i,hh,ww]
                                    idx += 1
                                end
                            end

                            z_sum = sum(z_vec)
                            @constraint(model, z_sum == 1)

                        elseif isa(layers_no_flatten[k], MeanPool) # calculate the average of the pixel values over the filter

                            # loop through each index in the previous layer that is under the filter
                            x_vec = vec([x[k-1,i,hh,ww] for hh in (h_filter*(h-1)+1):(h_filter*h), ww in (w_filter*(w-1)+1):(w_filter*w)])

                            temp_sum = sum(x_vec) / reduce(*, curr_filter_size)
                            @constraint(model, temp_sum == x[k,i,h,w])

                        end
                    end
                end
            end
        end
    end

    # loop through the Dense layers that are after the Flux.flatten non-indexed layer
    # in layer indices, we skip the flatten layer altogether (i.e., no variables at that layer)
    for k in K+1:K+D
        for node in 1:CNN_nodes[k+1][1]
            temp_sum = 0
            if k == K+1 # previous layer nodes still in matrix form, reshaping to a vector
                x_vec = vec([x[k-1,i,h,w] for h in 1:CNN_nodes[k][2], w in 1:CNN_nodes[k][3], i in 1:CNN_nodes[k][1]])
                temp_sum = sum(W[k][node,j] * x_vec[j] for j in 1:reduce(*, CNN_nodes[k]))
            else # previous layer nodes already in vector form, no reshaping needed
                temp_sum = sum(W[k][node,j] * x[k-1,j,1,1] for j in 1:reduce(*, CNN_nodes[k][1]))
            end
            if k < K+D # hidden Dense layer
                @constraint(model, temp_sum + b[k][node] == x[k,node,1,1] - s[k,node,1,1])
            else # output layer
                @constraint(model, temp_sum + b[k][node] == x[k,node,1,1])
            end
        end
    end

    # fix bounds to the hidden layers with relu activation function (Conv and Dense)
    for k in 1:length(layers_no_flatten)-1
        if isa(layers_no_flatten[k], Conv) || isa(layers_no_flatten[k], Dense)
            @constraint(model, [[k], i in 1:CNN_nodes[k+1][1], h in 1:CNN_nodes[k+1][2], w in 1:CNN_nodes[k+1][3]], 
                                x[k,i,h,w] <= U[k,i,h,w] * z[k,i,h,w])
            @constraint(model, [[k], i in 1:CNN_nodes[k+1][1], h in 1:CNN_nodes[k+1][2], w in 1:CNN_nodes[k+1][3]], 
                                s[k,i,h,w] <= -L[k,i,h,w] * (1 - z[k,i,h,w]))
        end
    end

    # arbitrary objective function (that can and should be changed later!) to allow optimization
    @objective(model, Max, x[1,1,1,1])

    return model

end

# inner function used in create_CNN_JuMP_model
# new img size after passing through 1) Conv layer filter or 2) a MaxPool / MeanPool layer
function next_sub_img_size(img::Tuple{Int64, Int64}, filter::Tuple{Int64, Int64}, pooling_layer::Bool = false)
    new_height = pooling_layer ? div(img[1], filter[1]) : (img[1] - filter[1] + 1)
    new_width  = pooling_layer ? div(img[2], filter[2]) : (img[2] - filter[2] + 1)
    return (new_height, new_width)
end

"""
evaluate_CNN!(CNN_model::Model, input::Array{Float32, 4})

Fixes the variables corresponding to the CNN input to a given input array.

# Arguments
- `CNN_model::Model`: A JuMP model representing a traied ReLU DNN (generated using the function create_JuMP_model).
- `input::Array{Float32, 4}`: A given input array to the trained CNN. 

# Examples
```julia
evaluate_CNN!(CNN_model, input)
```
"""
function evaluate_CNN!(CNN_model::Model, input::Array{Float32, 4})
    x = CNN_model[:x] # stores the @variable with name x from the JuMP_model
    len = size(input)
    for i in 1:len[3]
        for h in 1:len[1]
            for w in 1:len[2]
                fix(x[0,i,h,w], input[h,w,i,1], force=true)
            end
        end
    end
end
