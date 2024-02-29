# Public API

These are all of the functions and data structures that the user needs to know in order to use this package.

## Tree ensembles

### Data structures
* [`TEModel`](@ref) - holds the parameters from a tree ensemble model

### Tree model parameter extraction
* [`extract_evotrees_info`](@ref) - get the necessary parameters from an `EvoTrees` model

### MIP formulation
* [`TE_to_MIP`](@ref) - formulate a `JuMP` model from a tree ensemble without the split constraints

### Input optimization
* [`optimize_with_initial_constraints!`](@ref) - solve the tree ensemble optimization problem by creating all split constraints at the beginning
* [`optimize_with_lazy_constraints!`](@ref) - solve the tree ensemble optimization problem by creating the necessary split constraints as necessary

### Showing the solution
* [`get_solution`](@ref) - get human-readable solution in the form of upper and lower bounds for the input variables

## Neural networks

### Data structures
* [`SolverParams`](@ref) - holds the settings for each bound tightening solve

### MIP formulation
* [`NN_to_MIP_with_bound_tightening`](@ref) - formulate a `JuMP` model by performing simultaneous bound tightening
* [`NN_to_MIP_with_precomputed`](@ref) - formulate a `JuMP` model by utilizing precomputed neuron activation bounds in creating the *big-M* -constraints

### Compression
* [`compress_with_bound_tightening`](@ref) - perform compression by simulateous `JuMP` model construction and bound tightening
* [`compress_with_precomputed`](@ref) - perform compression with precomputed neuron activation bounds

### Forward pass
* [`forward_pass!`](@ref) - fix the input variables and optimize the model to get the output

## Convolutional neural networks

### Data structures
* [`CNNStructure`](@ref) - container for the layer stucture of a convolutional neural network model

### Parameter extraction
* [`get_structure`](@ref) - get layer structure from a convolutional neural network model

### MIP formulation
* [`create_MIP_from_CNN!`](@ref) - formulate a `JuMP` model from the CNN

### Forward pass
* [`image_pass!`](@ref) - fix the input variables and optimize the model to get the ouput