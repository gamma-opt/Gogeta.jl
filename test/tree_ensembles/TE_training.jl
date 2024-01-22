# This code has been used to generate the "paraboloid.bson" EvoTrees model.
# Now when testing the optimization, the tree ensemble doesn't have to be re-trained every time.

using EvoTrees

# Create a random sample of f(x,y,z)=x^2+y^2+z^2
x_train = rand(1000, 3) .- 0.5;
y_train = vec(sum(map.(x->x^2, x_train), dims=2));

# Train EvoTrees model
config = EvoTreeRegressor(nrounds=500, max_depth=5, nbins=16);
evo_model = fit_evotree(config; x_train, y_train);

# Extract splitpoints info
universal_tree_model = extract_evotrees_info(evo_model);

# Get averages between splitpoints
@views adjsum(a) = a[begin:end-1] + a[begin+1:end];
averages = Array{Vector}(undef, 3);
[averages[feat] = adjsum(universal_tree_model.splits_ordered[feat]) ./ 2 for feat in 1:3];

# Exhaustive search all possible inputs and find the minimum
# (the tree ensemble is a piecewise function so the search space is finite)
minimum = Vector{Float64}(undef, 3);
minimum_value = Inf64;
for x in averages[1], y in averages[2], z in averages[3]
    pred = EvoTrees.predict(evo_model, [x y z])[1]
    if pred < minimum_value
        minimum = [x, y, z]
        minimum_value = pred
    end
end

EvoTrees.save(evo_model, "paraboloid.bson")