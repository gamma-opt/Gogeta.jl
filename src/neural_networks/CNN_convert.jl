function create_model(CNN_model; big_M=1000.0)

    channels = Dict(0 => 1, 1 => 10, 2 => 10)
    dims = Dict(0 => 50, 1 => 48, 2 => 16)

    dense_lengths = Dict(0 => 2560, 1 => 100, 2 => 1)

    jump_model = Model(Gurobi.Optimizer)
    
    # 2-dimensional layers
    @variable(jump_model, c[layer=0:2, 1:dims[layer], 1:dims[layer], 1:channels[layer]] >= 0)
    @variable(jump_model, cs[layer=1:2, 1:dims[layer], 1:dims[layer], 1:channels[layer]] >= 0)
    @variable(jump_model, cz[layer=1:2, 1:dims[layer], 1:dims[layer], 1:channels[layer]], Bin)

    # dense layers
    @variable(jump_model, x[layer=0:2, 1:dense_lengths[layer]])
    @variable(jump_model, s[layer=1:1, 1:dense_lengths[layer]] >= 0)
    @variable(jump_model, z[layer=1:1, 1:dense_lengths[layer]], Bin)

    for (index, layer) in enumerate(CNN_model)
        if layer isa Conv
            println("Layer $index is convolutional")
            println("Filters: $(size(Flux.params(layer)[1]))")
            println("Biases: $(size(Flux.params(layer)[2]))")
            println("Pad: $(layer.pad)")
            println("Stride: $(layer.stride)")

            filters = [Flux.params(layer)[1][:, :, 1, i] for i in 1:size(layer(input))[3]]
            biases = Flux.params(layer)[2]

            filter_size = size(Flux.params(layer)[1])[1]
            filter_border = filter_size ÷ 2

            for x in 1:dims[index], y in 1:dims[index]

                # TODO include previous filter in formulas
                # TODO include pad and stride

                for filter in eachindex(filters)

                    convolution = @expression(jump_model, sum([filters[filter][filter_size+1-i, filter_size+1-j] * c[index-1, x-filter_border+i, y-filter_border+j, 1] for i in 1:filter_size, j in 1:filter_size]))
                    
                    @constraint(jump_model, c[index, x, y, filter] - cs[index, x, y, filter] == convolution + biases[filter])
                    @constraint(jump_model, c[index, x, y, filter] <= big_M * (1-cz[index, x, y, filter]))
                    @constraint(jump_model, cs[index, x, y, filter] <= big_M * cz[index, x, y, filter])
                end
            end

        elseif layer isa MaxPool
            println("Layer $index is maxpool")
            println("Size: $(layer.k)")
            println("Pad: $(layer.pad)")
            println("Stride: $(layer.stride)")

            filter_size = layer.k[1]
            filter_border = filter_size ÷ 2

            for x in 1:dims[index], y in 1:dims[index]
                # TODO multidimensional filters
                # TODO include pad and stride
                for image in 1:channels[index-1]
                    
                    @constraint(jump_model, [i in -1:1, j in -1:1], c[index, x, y, image] >= c[index-1, x*filter_size - filter_border + i, y*filter_size - filter_border + j, image])
                    
                    pz = @variable(jump_model, [-1:1, -1:1], Bin)
                    @constraint(jump_model, sum([pz[i, j] for i in -1:1, j in -1:1]) == 1)

                    @constraint(jump_model, [i in -1:1, j in -1:1], pz[i, j] => {c[index, x, y, image] <= c[index-1, x*filter_size - filter_border + i, y*filter_size - filter_border + j, image]})
                end
            end
        elseif layer == Flux.flatten
            println("Layer $index is flatten")
            println("Size of input: $(size(CNN_model[1:index-1](input)))")
            println("Size of output: $(size(CNN_model[1:index](input)))")

            @constraint(jump_model, [channel in 1:10, width in 1:16, height in 1:16], x[0, width + (height-1)*16 + (channel-1)*256] == c[index-1, width, height, channel])

        elseif layer isa Dense
            println("Layer $index is dense")
            println("Weights: $(size(Flux.params(layer)[1]))")
            println("Biases: $(size(Flux.params(layer)[2]))")

            weights = Flux.params(layer)[1]
            biases = Flux.params(layer)[2]
            layer = index-3

            n_neurons = length(biases)
            n_previous = length(x[layer-1, :])

            if CNN_model[index].σ == relu
                for neuron in 1:n_neurons
                    @constraint(jump_model, x[layer, neuron] >= 0)
                    @constraint(jump_model, x[layer, neuron] <= big_M * (1 - z[layer, neuron]))
                    @constraint(jump_model, s[layer, neuron] <= big_M * z[layer, neuron])
                    @constraint(jump_model, x[layer, neuron] - s[layer, neuron] == biases[neuron] + sum(weights[neuron, i] * x[layer-1, i] for i in 1:n_previous))
                end
            elseif CNN_model[index].σ == identity
                @constraint(jump_model, [neuron in 1:n_neurons], x[layer, neuron] == biases[neuron] + sum(weights[neuron, i] * x[layer-1, i] for i in 1:n_previous))
            end
        end

        println()
    end

    return jump_model
end

function image_pass!(jump_model, input, channel, layer)
    [fix(jump_model[:c][0, x, y, 1], input[x, y, 1, 1], force=true) for x in 1:50, y in 1:50]
    optimize!(jump_model)

    if layer == 1 return [value(jump_model[:c][1, x, y, channel]) for x in 1:48, y in 1:48]
    elseif layer == 2 return [value(jump_model[:c][2, x, y, channel]) for x in 1:16, y in 1:16]
    elseif layer == 3 return [value(jump_model[:x][0, n]) for n in 1:2560]
    elseif layer == 4 return [value(jump_model[:x][1, n]) for n in 1:100]
    elseif layer == 5 return [value(jump_model[:x][2, n]) for n in 1:1]
    end
end
