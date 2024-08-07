# Public API

These are all of the functions and data structures that the user needs to know in order to use this package.

## Tree ensembles

### Data structures
* [`TEModel`](@ref) - holds the parameters from a tree ensemble model

### Tree model parameter extraction
* [`extract_evotrees_info`](@ref) - get the necessary parameters from an `EvoTrees` model

### MIP formulation
* [`TE_formulate!`](@ref) - formulate a `JuMP` model from a tree ensemble without the split constraints

### Input optimization
* [`add_split_constraints!`](@ref) - add all split constraints to the formulation
* [`tree_callback_algorithm`](@ref) - used to add only necessary split constraints during callbacks

### Showing the solution
* [`get_solution`](@ref) - get human-readable solution in the form of upper and lower bounds for the input variables

## Neural networks

### MIP formulation
* [`NN_formulate!`](@ref) - formulate a `JuMP` model, perform simultaneous bound tightening and possibly compression
* [`NN_incorporate!`](@ref) - formulate a neural network MILP to be part of a larger `JuMP` model by linking the input and output variables

### Compression
* [`NN_compress`](@ref) - compress a neural network using precomputed activation bounds

### Forward pass
* [`forward_pass!`](@ref) - fix the input variables and optimize the model to get the output
* [`forward_pass_NN!`](@ref) - forward pass in a model with anonymous variables with the input and output variables given as arguments

### Sampling-based optimization
* [`optimize_by_sampling!`](@ref) - optimize the JuMP model by using a sampling-based approach
* [`optimize_by_walking!`](@ref) - optimize the JuMP model by using a more sophisticated sampling-based approach

## Input convex neural networks

### LP formulation
* [`ICNN_incorporate!`](@ref) - formulate an ICNN LP to be part of a larger `JuMP` model by linking the input and output variables

### Forward pass
* [`forward_pass_ICNN!`](@ref) - fix the input variables and optimize the model to get the output

### Feasibility
* [`check_ICNN`](@ref) - check whether the given inputs and outputs satisfy the ICNN


## Convolutional neural networks

### Data structures
* [`CNNStructure`](@ref) - container for the layer stucture of a convolutional neural network model

### Parameter extraction
* [`get_structure`](@ref) - get layer structure from a convolutional neural network model

### MIP formulation
* [`CNN_formulate!`](@ref) - formulate a `JuMP` model from the CNN

### Forward pass
* [`image_pass!`](@ref) - fix the input variables and optimize the model to get the ouput

### Sampling-based optimization 
* [`optimize_by_walking_CNN!`](@ref) - optimize the JuMP model by using a more sophisticated sampling-based approach