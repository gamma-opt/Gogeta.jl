"""
    function create_MIP_from_CNN!(jump_model::JuMP.Model, CNN_model::Flux.Chain, cnnstruct::CNNStructure)

Creates a mixed-integer optimization problem from a `Flux.Chain` convolutional neural network model.
The optimization formulation is saved in the `JuMP.Model` given as an input.

A dummy objective function of 1 is added to the model. The objective is left for the user to define.

The convolutional neural network must follow a certain structure:
- It must consist of (in order) convolutional and pooling layers, a `Flux.flatten` layer and finally dense layers
- I.e. allowed layer types: `Conv`, `MaxPool`, `MeanPool`, `Flux.flatten`, `Dense`
- The activation function for all of the convolutional layers and the dense layers must be `ReLU`
- The last dense layer must use the `identity` activation function
- Input size, filter size, stride and padding can be chosen freely

# Parameters
- `jump_model`: an empty optimization model where the formulation will be saved
- `CNN_model`: `Flux.Chain` containing the CNN
- `cnnstruct`: holds the layer structure of the CNN

"""
function CNN_formulate!(jump_model::JuMP.Model, CNN_model::Flux.Chain, cnnstruct::CNNStructure)

    empty!(jump_model)

    channels = cnnstruct.channels
    dims = cnnstruct.dims
    dense_lengths = cnnstruct.dense_lengths
    conv_inds = cnnstruct.conv_inds
    maxpool_inds = cnnstruct.maxpool_inds
    meanpool_inds = cnnstruct.meanpool_inds
    flatten_ind = cnnstruct.flatten_ind
    dense_inds  = cnnstruct.dense_inds
    
    # 2d layers
    @variable(jump_model, c[layer=union(0, conv_inds, maxpool_inds, meanpool_inds), 1:dims[layer][1], 1:dims[layer][2], 1:channels[layer]] >= 0) # input is always between 0 and 1
    @variable(jump_model, cs[layer=conv_inds, 1:dims[layer][1], 1:dims[layer][2], 1:channels[layer]] >= 0)
    @variable(jump_model, cz[layer=conv_inds, 1:dims[layer][1], 1:dims[layer][2], 1:channels[layer]], Bin)

    # dense layers
    @variable(jump_model, x[layer=union(flatten_ind, dense_inds), 1:dense_lengths[layer]])
    @variable(jump_model, s[layer=dense_inds[1:end-1], 1:dense_lengths[layer]] >= 0)
    @variable(jump_model, z[layer=dense_inds[1:end-1], 1:dense_lengths[layer]], Bin)

    U_bounds_dense = Dict{Int, Vector}()
    L_bounds_dense = Dict{Int, Vector}()

    pixel_or_pad(layer, row, col, channel) = if haskey(c, (layer, row, col, channel)) c[layer, row, col, channel] else 0.0 end

    for (layer_index, layer_data) in enumerate(CNN_model)

        if layer_index in conv_inds

            filters = [Flux.params(layer_data)[1][:, :, in_channel, out_channel] for in_channel in 1:channels[layer_index-1], out_channel in 1:channels[layer_index]]
            biases = Flux.params(layer_data)[2]

            f_height, f_width = size(filters[1, 1])

            for row in 1:dims[layer_index][1], col in 1:dims[layer_index][2]
                
                pos = (layer_data.stride[1]*(row-1) + 1 - layer_data.pad[1], layer_data.stride[2]*(col-1) + 1 - layer_data.pad[2])

                for out_channel in 1:channels[layer_index]

                    convolution = @expression(jump_model, 
                        sum([filters[in_channel, out_channel][f_height-i, f_width-j] * pixel_or_pad(layer_index-1, pos[1]+i, pos[2]+j, in_channel)
                        for i in 0:f_height-1, 
                            j in 0:f_width-1, 
                            in_channel in 1:channels[layer_index-1]])
                    )
                    
                    @constraint(jump_model, c[layer_index, row, col, out_channel] - cs[layer_index, row, col, out_channel] == convolution + biases[out_channel])
                    @constraint(jump_model, c[layer_index, row, col, out_channel] <= 1.0 * (1-cz[layer_index, row, col, out_channel]))
                    @constraint(jump_model, cs[layer_index, row, col, out_channel] <= 1.0 * cz[layer_index, row, col, out_channel])
                end
            end

        elseif layer_index in maxpool_inds

            p_height = layer_data.k[1]
            p_width = layer_data.k[2]

            for row in 1:dims[layer_index][1], col in 1:dims[layer_index][2]

                pos = (layer_data.stride[1]*(row-1) - layer_data.pad[1], layer_data.stride[2]*(col-1) - layer_data.pad[2])
                
                for channel in 1:channels[layer_index-1]
                    
                    @constraint(jump_model, [i in 1:p_height, j in 1:p_width], c[layer_index, row, col, channel] >= pixel_or_pad(layer_index-1, pos[1]+i, pos[2]+j, channel))
                    
                    pz = @variable(jump_model, [1:p_height, 1:p_width], Bin)
                    @constraint(jump_model, sum([pz[i, j] for i in 1:p_height, j in 1:p_width]) == 1)

                    @constraint(jump_model, [i in 1:p_height, j in 1:p_width], c[layer_index, row, col, channel] <= pixel_or_pad(layer_index-1, pos[1]+i, pos[2]+j, channel) + (1-pz[i, j]))
                end
            end

        elseif layer_index in meanpool_inds

            p_height = layer_data.k[1]
            p_width = layer_data.k[2]

            for row in 1:dims[layer_index][1], col in 1:dims[layer_index][2]

                pos = (layer_data.stride[1]*(row-1) - layer_data.pad[1], layer_data.stride[2]*(col-1) - layer_data.pad[2])
                
                for channel in 1:channels[layer_index-1]
                    @constraint(jump_model, c[layer_index, row, col, channel] == 1/(p_height*p_width) * sum(pixel_or_pad(layer_index-1, pos[1]+i, pos[2]+j, channel)
                        for i in 1:p_height, 
                            j in 1:p_width)
                    )
                end
            end

        elseif layer_index == flatten_ind

            @constraint(jump_model, [channel in 1:channels[layer_index-1], row in dims[layer_index-1][1]:-1:1, col in 1:dims[layer_index-1][2]], 
                x[flatten_ind, row + (col-1)*dims[layer_index-1][1] + (channel-1)*prod(dims[layer_index-1])] == c[layer_index-1, row, col, channel]
            )

        elseif layer_index in dense_inds
            
            weights = Flux.params(layer_data)[1]
            biases = Flux.params(layer_data)[2]

            n_neurons = length(biases)
            n_previous = length(x[layer_index-1, :])

            # compute heuristic bounds
            if layer_index == minimum(dense_inds)
                U_bounds_dense[layer_index] = [sum(max(weights[neuron, previous] * 1.0, 0.0) for previous in 1:n_previous) + biases[neuron] for neuron in 1:n_neurons]
                L_bounds_dense[layer_index] = [sum(min(weights[neuron, previous] * 1.0, 0.0) for previous in 1:n_previous) + biases[neuron] for neuron in 1:n_neurons]
            else
                U_bounds_dense[layer_index] = [sum(max(weights[neuron, previous] * max(0, U_bounds_dense[layer_index-1][previous]), weights[neuron, previous] * max(0, L_bounds_dense[layer_index-1][previous])) for previous in 1:n_previous) + biases[neuron] for neuron in 1:n_neurons]
                L_bounds_dense[layer_index] = [sum(min(weights[neuron, previous] * max(0, U_bounds_dense[layer_index-1][previous]), weights[neuron, previous] * max(0, L_bounds_dense[layer_index-1][previous])) for previous in 1:n_previous) + biases[neuron] for neuron in 1:n_neurons]
            end

            if layer_data.σ == relu
                for neuron in 1:n_neurons
                    @constraint(jump_model, x[layer_index, neuron] >= 0)
                    @constraint(jump_model, x[layer_index, neuron] <= U_bounds_dense[layer_index][neuron] * (1 - z[layer_index, neuron]))
                    @constraint(jump_model, s[layer_index, neuron] <= -L_bounds_dense[layer_index][neuron] * z[layer_index, neuron])
                    @constraint(jump_model, x[layer_index, neuron] - s[layer_index, neuron] == biases[neuron] + sum(weights[neuron, i] * x[layer_index-1, i] for i in 1:n_previous))
                end
            elseif layer_data.σ == identity
                @constraint(jump_model, [neuron in 1:n_neurons], x[layer_index, neuron] == biases[neuron] + sum(weights[neuron, i] * x[layer_index-1, i] for i in 1:n_previous))
            end
        end
    end

    @objective(jump_model, Max, 1)
end
