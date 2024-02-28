using Flux
using Plots
using JuMP
using Gurobi
using Random
using Gogeta

using Images
using FileIO
image = load("/Users/eetureijonen/Pictures/IMG_0195.JPG"); # swap in your own image
downscaled_image = imresize(image, (70, 50));

input = reshape(Float32.(channelview(Gray.(downscaled_image))), 70, 50, 1, 1);
input = input[end:-1:1, :, :, :];
size(CNN_model[1:6](input))

Random.seed!(1234)
CNN_model = Flux.Chain(
    Conv((4,3), 1 => 10, pad=(2, 1), stride=(3, 2), relu),
    MeanPool((5,3), pad=(3, 2), stride=(2, 2)),
    MaxPool((3,4), pad=(1, 3), stride=(3, 2)),
    Conv((4,3), 10 => 5, pad=(2, 1), stride=(3, 2), relu),
    MaxPool((3,4), pad=(1, 3), stride=(3, 2)),
    Flux.flatten,
    Dense(20 => 100, relu),
    Dense(100 => 1)
)

# input image
heatmap(input[:, :, 1, 1], background=false, legend=false, color=:inferno, aspect_ratio=:equal, axis=([], false))

# convolution layer outputs
outputs = [CNN_model[1](input)[:, :, channel, 1] for channel in 1:10];
display.(heatmap.(outputs, background=false, legend=false, color = :inferno, aspect_ratio=:equal, axis=([], false)));

# meanpool outputs
outputs = [CNN_model[1:2](input)[:, :, channel, 1] for channel in 1:10];
display.(heatmap.(outputs, background=false, legend=false, color = :inferno, aspect_ratio=:equal, axis=([], false)));

# maxpool
outputs = [CNN_model[1:3](input)[:, :, channel, 1] for channel in 1:10];
display.(heatmap.(outputs, background=false, legend=false, color = :inferno, aspect_ratio=:equal, axis=([], false)));

# new conv
outputs = [CNN_model[1:4](input)[:, :, channel, 1] for channel in 1:5];
display.(heatmap.(outputs, background=false, legend=false, color = :inferno, aspect_ratio=:equal, axis=([], false)));

# last maxpool
outputs = [CNN_model[1:5](input)[:, :, channel, 1] for channel in 1:5];
display.(heatmap.(outputs, background=false, legend=false, color = :inferno, aspect_ratio=:equal, axis=([], false)));

# create jump model from cnn
jump = Model(Gurobi.Optimizer)
set_silent(jump)
cnns = get_structure(CNN_model, input);
create_MIP_from_CNN!(jump, CNN_model, cnns)

# Test that jump model produces same outputs for all layers as the CNN
@time CNN_model[1](input)[:, :, :, 1] ≈ image_pass!(jump, input, cnns, 1)
@time CNN_model[1:2](input)[:, :, :, 1] ≈ image_pass!(jump, input, cnns, 2)
@time CNN_model[1:3](input)[:, :, :, 1] ≈ image_pass!(jump, input, cnns, 3)
@time CNN_model[1:4](input)[:, :, :, 1] ≈ image_pass!(jump, input, cnns, 4)
@time CNN_model[1:5](input)[:, :, :, 1] ≈ image_pass!(jump, input, cnns, 5)
@time vec(CNN_model[1:6](input)) ≈ image_pass!(jump, input, cnns, 6)
@time vec(CNN_model[1:7](input)) ≈ image_pass!(jump, input, cnns, 7)
@time vec(CNN_model[1:8](input)) ≈ image_pass!(jump, input, cnns, 8)
@time vec(CNN_model(input)) ≈ image_pass!(jump, input)

# Plot true model maxpool fifth channel
heatmap(CNN_model[1:2](input)[:, :, 5, 1], background=false, legend=false, color=:inferno, aspect_ratio=:equal, axis=([], false))

# Plot jump model maxpool fifth channel
heatmap(image_pass!(jump, input, cnns, 2)[:, :, 5], background=false, legend=false, color=:inferno, aspect_ratio=:equal, axis=([], false))

