@kwdef struct CNNStructure
end

function create_model!(jump_model, CNN_model; big_M=1000.0)

    channels = Dict(0 => 1, 1 => 10, 2 => 10)
    dims = Dict(0 => (50, 50), 1 => (48, 48), 2 => (16, 16))

    dense_lengths = Dict(0 => 2560, 1 => 100, 2 => 1)

    conv_inds = [1]
    maxpool_inds = [2]
    flatten_ind = [3]
    dense_inds = [4, 5]

    n_conv_or_pool = length(conv_inds) + length(maxpool_inds)
    n_dense = length(dense_inds)
    
    # 2-dimensional layers
    @variable(jump_model, c[layer=0:n_conv_or_pool, 1:dims[layer][1], 1:dims[layer][1], 1:channels[layer]] >= 0) # input is always between 0 and 1
    @variable(jump_model, cs[layer=1:n_conv_or_pool, 1:dims[layer][1], 1:dims[layer][2], 1:channels[layer]] >= 0) # TODO correct index sets
    @variable(jump_model, cz[layer=1:n_conv_or_pool, 1:dims[layer][1], 1:dims[layer][2], 1:channels[layer]], Bin)

    # dense layers
    @variable(jump_model, x[layer=0:n_dense, 1:dense_lengths[layer]])
    @variable(jump_model, s[layer=1:n_dense-1, 1:dense_lengths[layer]] >= 0)
    @variable(jump_model, z[layer=1:n_dense-1, 1:dense_lengths[layer]], Bin)

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

            @constraint(jump_model, [channel in 1:channels[index-1], row in 1:dims[index-1][1], col in 1:dims[index-1][2]], x[0, row + (col-1)*dims[index-1][2] + (channel-1)*prod(dims[index-1])] == c[index-1, row, col, channel])

        elseif layer_data isa Dense
            println("Layer $index is dense")
            println("Weights: $(size(Flux.params(layer_data)[1]))")
            println("Biases: $(size(Flux.params(layer_data)[2]))")

            weights = Flux.params(layer_data)[1]
            biases = Flux.params(layer_data)[2]
            dense_index = index-3

            n_neurons = length(biases)
            n_previous = length(x[dense_index-1, :])

            if layer_data.σ == relu
                for neuron in 1:n_neurons
                    @constraint(jump_model, x[dense_index, neuron] >= 0)
                    @constraint(jump_model, x[dense_index, neuron] <= big_M * (1 - z[dense_index, neuron]))
                    @constraint(jump_model, s[dense_index, neuron] <= big_M * z[dense_index, neuron])
                    @constraint(jump_model, x[dense_index, neuron] - s[dense_index, neuron] == biases[neuron] + sum(weights[neuron, i] * x[dense_index-1, i] for i in 1:n_previous))
                end
            elseif layer_data.σ == identity
                @constraint(jump_model, [neuron in 1:n_neurons], x[dense_index, neuron] == biases[neuron] + sum(weights[neuron, i] * x[dense_index-1, i] for i in 1:n_previous))
            end
        end

        println()
    end
end

function image_pass!(jump_model, input, layer)
    [fix(jump_model[:c][0, x, y, channel], input[x, y, channel, 1], force=true) for x in 1:50, y in 1:50, channel in 1:1]
    optimize!(jump_model)

    if layer == 1 return [value(jump_model[:c][1, x, y, channel]) for x in 1:48, y in 1:48, channel in 1:10]
    elseif layer == 2 return [value(jump_model[:c][2, x, y, channel]) for x in 1:16, y in 1:16, channel in 1:10]
    elseif layer == 3 return [value(jump_model[:x][0, n]) for n in 1:2560]
    elseif layer == 4 return [value(jump_model[:x][1, n]) for n in 1:100]
    elseif layer == 5 return [value(jump_model[:x][2, n]) for n in 1:1]
    end
end
