import Plots as pl

filepath = string(@__DIR__)*"/test_results/concrete_test_results.txt";
filepath = string(@__DIR__)*"/test_results/OX2_test_results.txt";

stores = Dict([(3, 1), (5, 2), (7, 3), (9, 4), (12, 5)]);

r2_test_values = Array{Array}(undef, 5);
[r2_test_values[depth] = [] for depth in 1:5];

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

        println("$(parse(Int, chop(split(line, " ")[4]))), $(parse(Int, chop(split(line, " ")[6]))), $(parse(Float64, chop(split(line, " ")[9]))), $(parse(Float64, chop(split(line, " ")[12])))")
    elseif length(split(line, " ")) == 9 # lines with training time data
        println("")
        println("$(parse(Int, chop(split(line, " ")[4]))), $(parse(Int, chop(split(line, " ")[6]))), $(parse(Float64, chop(split(line, " ")[9])))")
    end

end

pl.plot(    trees, 
            r2_test_values,
            marker=4, 
            ylim=[0.8, 1], 
            ylabel="Coeffient of determination",
            xlabel="Number of trees",
            label=["Depth 3" "Depth 5" "Depth 7" "Depth 9" "Depth 12"],
            title="Model quality"
        )