"""
    struct CNNStructure

Container for the layer structure of a convolutional neural network.

This structure is used for passing the CNN parameters from one function to another.
This structure can be created with the `get_structure` -function.

# Fields
- `channels`: dictionary of (layer, number of channels) pairs for convolutional or pooling layers
- `dims`: dictionary of (layer, (# of rows, # of columns)) pairs for convolutional or pooling layers
- `dense_lengths`: dictionary of (layer, # of neurons) pairs for dense and flatten layers
- `conv_inds`: vector of the convolutional layer indices
- `maxpool_inds_inds`: vector of the maxpool layer indices
- `meanpool_inds`: vector of the meanpool layer indices
- `flatten_ind`: index of the `Flux.flatten` layer
- `dense_inds`: vector of the dense layer indices

```
"""
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

"""
    function get_structure(CNN_model::Flux.Chain, input::Array{Float32, 4})

Extract the layer structure of a convolutional neural network.
The input image is needed to calculate the correct sizes for the hidden 2-dimensional layers.

Returns a `CNNStructure` struct.
"""
function get_structure(CNN_model::Flux.Chain, input::Array{Float32, 4})

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

"""
    function image_pass!(jump_model::JuMP.Model, input::Array{Float32, 4}, cnnstruct::CNNStructure, layer::Int)

**Debugging version**

Forward pass an image through the JuMP model representing a convolutional neural network.

Returns the output of the layer with index given as input.
"""
function image_pass!(jump_model::JuMP.Model, input::Array{Float32, 4}, cnnstruct::CNNStructure, layer::Int)
    [fix(jump_model[:c][0, row, col, channel], input[row, col, channel, 1], force=true) for row in 1:cnnstruct.dims[0][1], col in 1:cnnstruct.dims[0][2], channel in 1:cnnstruct.channels[0]]
    optimize!(jump_model)

    if layer in union(0, cnnstruct.conv_inds, cnnstruct.maxpool_inds, cnnstruct.meanpool_inds) 
        return [value(jump_model[:c][layer, row, col, channel]) for row in 1:cnnstruct.dims[layer][1], col in 1:cnnstruct.dims[layer][2], channel in 1:cnnstruct.channels[layer]]
    elseif layer in union(cnnstruct.dense_inds, cnnstruct.flatten_ind)
        return [value(jump_model[:x][layer, neuron]) for neuron in 1:cnnstruct.dense_lengths[layer]]
    end
end

"""
    function image_pass!(jump_model::JuMP.Model, input::Array{Float32, 4})

Forward pass an image through the JuMP model representing a convolutional neural network.

Returns the output of the network, i.e., a vector of the activations of the last dense layer neurons.
"""
function image_pass!(jump_model::JuMP.Model, input::Array{Float32, 4})
    
    [fix(jump_model[:c][0, row, col, channel], input[row, col, channel, 1], force=true) for row in eachindex(input[:, 1, 1, 1]), col in eachindex(input[1, :, 1, 1]), channel in eachindex(input[1, 1, :, 1])]
    optimize!(jump_model)

    (last_layer, outputs) = maximum(keys(jump_model[:x].data))
    result = [value(jump_model[:x][last_layer, i]) for i in 1:outputs]
    unfix.(jump_model[:c][0, :, :, :])

    return result
end