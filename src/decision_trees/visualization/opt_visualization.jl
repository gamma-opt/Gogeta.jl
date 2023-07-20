import Plots as pl

dataset = "concrete"
parent_directory = "/Users/eetureijonen/Desktop/THESIS/ML_as_MO/src/decision_trees/";

filepath = parent_directory*"/test_results/"*dataset*"_opt_results.txt";

filternums(text) = try parse(Float64, text) catch error NaN end;
index = Dict([(3, 1), (5, 2), (7, 3), (9, 4), (12, 5)]);

elements = dataset == "3A4" ? 4 : 5;

solution_times = Array{Array}(undef, elements);
[solution_times[depth] = [] for depth in 1:elements];
trees = [];

for line in eachline(filepath)
    
    numbers = filter(num->!isnan(num), filternums.(replace.(string.(split(line, " ")), (","=>""))))

    if length(numbers) == 12

        n_trees = Int.(numbers[1])
        depth = Int.(numbers[2])

        pre_time_normal = numbers[3]
        opt_time_normal = numbers[4]

        pre_time_alg = numbers[6]
        opt_time_alg = numbers[7]

        n_levels = Int.(numbers[9])
        n_leaves = Int.(numbers[10])

        init_cons = Int.(numbers[11])
        added_cons = Int.(numbers[12])

        if (n_trees in trees) == false
            push!(trees, n_trees) 
        end
    
        push!(solution_times[index[depth]], opt_time_normal)

    end
end

display(
pl.plot(    trees, 
            palette=:rainbow,
            solution_times,
            markershape=:xcross, 
            #yaxis=:log,
            legend=:topleft,
            #yticks=([0.1, 1, 10, 100, 1000], string.([0.1, 1, 10, 100, 1000])),
            ylabel="Solution time (seconds)",
            xlabel="Number of trees",
            label=["Depth 3" "Depth 5" "Depth 7" "Depth 9" "Depth 12"],
            title="Optimization performance for "*dataset*" dataset"
        )
)