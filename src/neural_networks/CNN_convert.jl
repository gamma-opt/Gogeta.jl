function create_model(CNN_model)
    channels(layer) = if layer == 0 1 elseif layer == 1 10 elseif layer == 2 10 end
    sizes(layer) = if layer == 0 50 elseif layer == 1 48 elseif layer == 2 16 end

    jump_model = Model(Gurobi.Optimizer)
    @variable(jump_model, c[layer=0:2, x=1:sizes(layer), y=1:sizes(layer), channel=1:channels(layer)] >= 0)
    @variable(jump_model, cs[layer=1:2, x=1:sizes(layer), y=1:sizes(layer), channel=1:channels(layer)] >= 0)
    @variable(jump_model, cz[layer=1:2, x=1:sizes(layer), y=1:sizes(layer), channel=1:channels(layer)], Bin)
    @variable(jump_model, x[layer=0:1, if layer == 0 neuron=1:2560 else neuron=1:10 end])

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

            for x in 1:sizes(index), y in 1:sizes(index)

                # TODO include previous filter in formulas
                # TODO include pad and stride

                for filter in eachindex(filters)

                    convolution = @expression(jump_model, sum([filters[filter][filter_size+1-i, filter_size+1-j] * c[index-1, x-filter_border+i, y-filter_border+j, 1] for i in 1:filter_size, j in 1:filter_size]))
                    
                    @constraint(jump_model, c[index, x, y, filter] - cs[index, x, y, filter] == convolution + biases[filter])
                    @constraint(jump_model, c[index, x, y, filter] <= 1000.0 * (1-cz[index, x, y, filter]))
                    @constraint(jump_model, cs[index, x, y, filter] <= 1000.0 * cz[index, x, y, filter])
                end
            end

        elseif layer isa MaxPool
            println("Layer $index is maxpool")
            println("Size: $(layer.k)")
            println("Pad: $(layer.pad)")
            println("Stride: $(layer.stride)")

            filter_size = layer.k[1]
            filter_border = filter_size ÷ 2

            for x in 1:sizes(index), y in 1:sizes(index)
                # TODO include previous filter in formulas
                # TODO include pad and stride
                for image in 1:channels(index-1)
                    
                    @constraint(jump_model, [i in -1:1, j in -1:1], c[index, x, y, image] >= c[index-1, x*filter_size - filter_border + i, y*filter_size - filter_border + j, image])
                    
                    z = @variable(jump_model, [-1:1, -1:1], Bin)
                    @constraint(jump_model, sum([z[i, j] for i in -1:1, j in -1:1]) == 1)

                    @constraint(jump_model, [i in -1:1, j in -1:1], z[i, j] => {c[index, x, y, image] <= c[index-1, x*filter_size - filter_border + i, y*filter_size - filter_border + j, image]})
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
        end
        println()
    end

    return jump_model
end

function image_pass!(jump_model, input, index, layer)
    [fix(jump_model[:c][0, x, y, 1], input[x, y, 1, 1], force=true) for x in 1:50, y in 1:50]
    optimize!(jump_model)
    if layer == 1 return [value(jump_model[:c][1, x, y, index]) for x in 1:48, y in 1:48]
    elseif layer == 2 return [value(jump_model[:c][2, x, y, index]) for x in 1:16, y in 1:16]
    elseif layer == 3 return [value(jump_model[:x][0, n]) for n in 1:2560]
    end
end
