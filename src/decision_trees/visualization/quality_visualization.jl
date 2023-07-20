import Plots as pl
using LaTeXStrings

datasets = ["concrete", "OX2", "3A4"];
stores = Dict([(3, 1), (5, 2), (7, 3), (9, 4), (12, 5)]);

for dataset in datasets

filepath = string(@__DIR__)*"/test_results/"*dataset*"_test_results.txt";

elements = dataset == "3A4" ? 4 : 5;

r2_test_values = Array{Array}(undef, elements);
[r2_test_values[depth] = [] for depth in 1:elements];

#r2_train_values = Array{Array}(undef, 5);
#[r2_train_values[depth] = [] for depth in 1:5];

trees = [10, 50, 100, 200, 350, 500, 750, 1000];

for line in eachline(filepath)

    if length(split(line, " ")) == 12 # lines with model qualty data

        n_trees = parse(Int, chop(split(line, " ")[4]));
        depth = parse(Int, chop(split(line, " ")[6]));
        r2_train = parse(Float64, chop(split(line, " ")[9]));
        r2_test = parse(Float64, chop(split(line, " ")[12]));

        push!(r2_test_values[stores[depth]], r2_test)

    elseif length(split(line, " ")) == 9 # lines with training time data

        n_trees = parse(Int, chop(split(line, " ")[4]));
        depth = parse(Int, chop(split(line, " ")[6]));
        train_time = parse(Float64, chop(split(line, " ")[9]));

    end

end

display(
pl.plot(    trees, 
            palette=:rainbow,
            r2_test_values,
            markershape=:xcross, 
            ylim=[minimum(vcat(r2_test_values...)) - .05, maximum(vcat(r2_test_values...)) + .05], 
            ylabel=L"R^{2}"*" for test data",
            xlabel="Number of trees",
            label=["Depth 3" "Depth 5" "Depth 7" "Depth 9" "Depth 12"],
            title="Model quality for dataset "*dataset
        )
)

end