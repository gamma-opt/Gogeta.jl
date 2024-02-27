function create_model!(jump_model, CNN_model, input; big_M=1000.0)

    cnnstruct = get_structure(CNN_model, input)

    channels = cnnstruct.channels
    dims = cnnstruct.dims
    dense_lengths = cnnstruct.dense_lengths
    conv_inds = cnnstruct.conv_inds
    maxpool_inds = cnnstruct.maxpool_inds
    flatten_ind = cnnstruct.flatten_ind
    dense_inds  = cnnstruct.dense_inds
    
    # 2d layers
    @variable(jump_model, c[layer=union(0, conv_inds, maxpool_inds), 1:dims[layer][1], 1:dims[layer][2], 1:channels[layer]] >= 0) # input is always between 0 and 1
    @variable(jump_model, cs[layer=conv_inds, 1:dims[layer][1], 1:dims[layer][2], 1:channels[layer]] >= 0)
    @variable(jump_model, cz[layer=conv_inds, 1:dims[layer][1], 1:dims[layer][2], 1:channels[layer]], Bin)

    # dense layers
    @variable(jump_model, x[layer=union(flatten_ind, dense_inds), 1:dense_lengths[layer]])
    @variable(jump_model, s[layer=dense_inds[1:end-1], 1:dense_lengths[layer]] >= 0)
    @variable(jump_model, z[layer=dense_inds[1:end-1], 1:dense_lengths[layer]], Bin)

    for (index, layer_data) in enumerate(CNN_model)

        if layer_data isa Conv
            println("Layer $index is convolutional")
            println("Filters: $(size(Flux.params(layer_data)[1]))")
            println("Biases: $(size(Flux.params(layer_data)[2]))")
            println("Pad: $(layer_data.pad)")
            println("Stride: $(layer_data.stride)")

            filters = [Flux.params(layer_data)[1][:, :, in_channel, out_channel] for in_channel in 1:channels[index-1], out_channel in 1:channels[index]]
            biases = Flux.params(layer_data)[2]

            f_height = size(Flux.params(layer_data)[1])[1]
            f_width = size(Flux.params(layer_data)[1])[2]

            for row in 1:dims[index][1], col in 1:dims[index][2]

                # TODO include pad and stride

                for out_channel in 1:channels[index]

                    convolution = @expression(jump_model, sum([filters[in_channel, out_channel][f_height-i, f_width-j] * c[index-1, row+i, col+j, in_channel] for i in 0:f_height-1, j in 0:f_width-1, in_channel in 1:channels[index-1]])) # TODO think about this calculation
                    
                    @constraint(jump_model, c[index, row, col, out_channel] - cs[index, row, col, out_channel] == convolution + biases[out_channel])
                    @constraint(jump_model, c[index, row, col, out_channel] <= big_M * (1-cz[index, row, col, out_channel]))
                    @constraint(jump_model, cs[index, row, col, out_channel] <= big_M * cz[index, row, col, out_channel])
                end
            end

        elseif layer_data isa MaxPool
            println("Layer $index is maxpool")
            println("Size: $(layer_data.k)")
            println("Pad: $(layer_data.pad)")
            println("Stride: $(layer_data.stride)")

            p_height = layer_data.k[1]
            p_width = layer_data.k[2]

            for row in 1:dims[index][1], col in 1:dims[index][2]

                # TODO include pad and stride
                
                for channel in 1:channels[index-1]
                    
                    @constraint(jump_model, [i in 1:p_height, j in 1:p_width], c[index, row, col, channel] >= c[index-1, (row-1)*p_height + i, (col-1)*p_width + j, channel])  # TODO think about this calculation
                    
                    pz = @variable(jump_model, [1:p_height, 1:p_width], Bin)
                    @constraint(jump_model, sum([pz[i, j] for i in 1:p_height, j in 1:p_width]) == 1)

                    @constraint(jump_model, [i in 1:p_height, j in 1:p_width], pz[i, j] => {c[index, row, col, channel] <= c[index-1, (row-1)*p_height + i, (col-1)*p_width + j, channel]})
                end
            end
        elseif layer_data == Flux.flatten
            println("Layer $index is flatten")
            println("Size of input: $(size(CNN_model[1:index-1](input)))")
            println("Size of output: $(size(CNN_model[1:index](input)))")

            @constraint(jump_model, [channel in 1:channels[index-1], row in 1:dims[index-1][1], col in 1:dims[index-1][2]], x[flatten_ind, row + (col-1)*dims[index-1][2] + (channel-1)*prod(dims[index-1])] == c[index-1, row, col, channel])

        elseif layer_data isa Dense
            println("Layer $index is dense")
            println("Weights: $(size(Flux.params(layer_data)[1]))")
            println("Biases: $(size(Flux.params(layer_data)[2]))")

            weights = Flux.params(layer_data)[1]
            biases = Flux.params(layer_data)[2]

            n_neurons = length(biases)
            n_previous = length(x[index-1, :])

            if layer_data.σ == relu
                for neuron in 1:n_neurons
                    @constraint(jump_model, x[index, neuron] >= 0)
                    @constraint(jump_model, x[index, neuron] <= big_M * (1 - z[index, neuron]))
                    @constraint(jump_model, s[index, neuron] <= big_M * z[index, neuron])
                    @constraint(jump_model, x[index, neuron] - s[index, neuron] == biases[neuron] + sum(weights[neuron, i] * x[index-1, i] for i in 1:n_previous))
                end
            elseif layer_data.σ == identity
                @constraint(jump_model, [neuron in 1:n_neurons], x[index, neuron] == biases[neuron] + sum(weights[neuron, i] * x[index-1, i] for i in 1:n_previous))
            end
        end

        println()
    end
end
