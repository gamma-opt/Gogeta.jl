# Example of using the implementation of the MIP formulation for the gradient boosted decision trees assembly

using XLSX
using EvoTrees
using Random
using Statistics

# Read a concrete data isntance 
xf = XLSX.readxlsx("examples/data/Concrete_Data.xlsx")
data = Float64.(xf["Sheet1"]["A2:I1031"])

# generate traininig data for the EveTree boosted trees model formulation
Random.seed!(1)
data = data[shuffle(1:end), :]

split::Int = floor(0.75 * length(data[:, 1]))

#Split the data into training and testing batches
x_train = data[1:split, 1:8];
y_train = data[1:split, 9];

x_test = data[split+1:end, 1:8];
y_test = data[split+1:end, 9];

# create and train EvoTree boosted trees model 
config = EvoTreeRegressor(nrounds=1000, max_depth=5);
model = fit_evotree(config; x_train, y_train);

pred_train = EvoTrees.predict(model, x_train);
pred_test = EvoTrees.predict(model, x_test);

using JuMP
using Gurobi
using Gogeta

# tranform the EvoTree model to TEModel that is the inout of the tree_model_to_MIP function 
new_model = extract_evotrees_info(model)

# Create the MIP problem given the input TEModel and consdering all the contraints are added to the model in the beginning
x_new, sol_new, m_new = tree_model_to_MIP(new_model, create_initial = true)

# Create the MIP problem given the input TEModel and consdering no contraints to be added to the model in the beginning and constraints generation algorithm to be used (using lazy constraints)
x_alg, sol_alg, m_algo = tree_model_to_MIP(new_model, create_initial = false)

println("Predicition of EvoTrees model given the otimal solution of correspoding MIP problem as an input")
println("Prediction of EvoTrees model for the solution of MIP problem solved without lazy constraints algorithm: $(EvoTrees.predict(model, reshape([mean(x_new[n]) for n in 1:8], 1, 8))[1])")
println("Prediction of EvoTrees model for the solution of MIP problem solved with lazy constraints algorithm: $(EvoTrees.predict(model, reshape([mean(x_alg[n]) for n in 1:8], 1, 8))[1])")
println("Maximum conceivable sum of tree predictions: $(sum(maximum(model.trees[tree].pred) for tree in 1:1001))")
println("Optimal objective value of the MIP problem solved without lazy constraints algorithm: $sol_new")
println("Optimal objective value of the MIP problem solved with lazy constraints algorithm: $sol_alg")
println("Maximum of the test dataset: $(maximum(pred_test))")