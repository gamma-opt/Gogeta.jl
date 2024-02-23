using Flux
using Plots
using JuMP
using Gurobi

include("CNN_convert.jl")

CNN_model = Flux.Chain(
    Conv((3,3), 1 => 10, relu),
    MaxPool((3,3)),
    Flux.flatten,
    Dense(640 => 100, relu),
    Dense(100 => 1)
)

using Images
using FileIO
image = load("/Users/eetureijonen/Pictures/IMG_0195.JPG");
downscaled_image = imresize(image, (50, 50));

input = reshape(Float32.(channelview(Gray.(downscaled_image))), 50, 50, 1, 1);
input = input[end:-1:1, :, :, :];
heatmap(input[:, :, 1, 1], background=false, legend=false, color=:inferno, aspect_ratio=:equal, axis=([], false))

outputs = [CNN_model[1](input)[:, :, i, 1] for i in 1:size(CNN_model[1:2](input))[3]];
display.(heatmap.(outputs, background=false, legend=false, color = :inferno, aspect_ratio=:equal, axis=([], false)));

jump = create_model(CNN_model)
out = image_pass!(jump, input, 3);

CNN_model[1](input)[:, :, 1, 1] ≈ out

heatmap(CNN_model[1](input)[:, :, 3, 1], background=false, legend=false, color=:inferno, aspect_ratio=:equal, axis=([], false))
heatmap(out, background=false, legend=false, color=:inferno, aspect_ratio=:equal, axis=([], false))

outputs = [CNN_model[1:2](input)[:, :, i, 1] for i in 1:size(CNN_model[1:2](input))[3]];
display.(heatmap.(outputs, background=false, legend=false, color = :inferno, aspect_ratio=:equal, axis=([], false)));

