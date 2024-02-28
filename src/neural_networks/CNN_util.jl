struct CNNStructure
    channels::Dict{Int, Int}
    dims::Dict{Int64, Tuple{Int64, Int64}}
    dense_lengths::Dict{Int, Int}
    conv_inds::Vector{Int}
    maxpool_inds::Vector{Int}
    meanpool_inds::Vector{Int}
    flatten_ind::Int
    dense_inds::Vector{Int}
end

function get_structure(CNN_model, input)

    channels = Dict{Int, Int}()
    dims = Dict{Int64, Tuple{Int64, Int64}}()
    dense_lengths = Dict{Int, Int}()
    conv_inds = Vector{Int}()
    maxpool_inds = Vector{Int}()
    meanpool_inds = Vector{Int}()
    flatten_ind::Int = -1
    dense_inds = Vector{Int}()

    push!(channels, 0 => size(input)[3])
    push!(dims, 0 => size(input)[1:2])

    for (layer_index, layer_data) in enumerate(CNN_model)
         
        if layer_data isa Conv
            w, h, c, _ = size(CNN_model[1:layer_index](input))
            push!(channels, layer_index => c)
            push!(conv_inds, layer_index)
            push!(dims, layer_index => (w, h))

        elseif layer_data isa MaxPool
            w, h, c, _ = size(CNN_model[1:layer_index](input))
            push!(channels, layer_index => c)
            push!(maxpool_inds, layer_index)
            push!(dims, layer_index => (w, h))

        elseif layer_data isa MeanPool
            w, h, c, _ = size(CNN_model[1:layer_index](input))
            push!(channels, layer_index => c)
            push!(meanpool_inds, layer_index)
            push!(dims, layer_index => (w, h))

        elseif layer_data == Flux.flatten
            flatten_ind = layer_index
            push!(dense_lengths, layer_index => size(CNN_model[1:layer_index](input))[1])

        elseif layer_data isa Dense
            push!(dense_inds, layer_index)
            push!(dense_lengths, layer_index => size(CNN_model[1:layer_index](input))[1])
        end

    end
    
    return CNNStructure(channels, dims, dense_lengths, conv_inds, maxpool_inds, meanpool_inds, flatten_ind, dense_inds)
end

function image_pass!(jump_model, input, cnnstruct, layer)
    [fix(jump_model[:c][0, row, col, channel], input[row, col, channel, 1], force=true) for row in 1:cnnstruct.dims[0][1], col in 1:cnnstruct.dims[0][2], channel in 1:cnnstruct.channels[0]]
    optimize!(jump_model)

    if layer in union(0, cnnstruct.conv_inds, cnnstruct.maxpool_inds, cnnstruct.meanpool_inds) 
        return [value(jump_model[:c][layer, row, col, channel]) for row in 1:cnnstruct.dims[layer][1], col in 1:cnnstruct.dims[layer][2], channel in 1:cnnstruct.channels[layer]]
    elseif layer in union(cnnstruct.dense_inds, cnnstruct.flatten_ind)
        return [value(jump_model[:x][layer, neuron]) for neuron in 1:cnnstruct.dense_lengths[layer]]
    end
end