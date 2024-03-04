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

### Compression
* [`NN_compress`](@ref) - compress a neural network using precomputed activation bounds

### Forward pass
* [`forward_pass!`](@ref) - fix the input variables and optimize the model to get the output

## Convolutional neural networks

### Data structures
* [`CNNStructure`](@ref) - container for the layer stucture of a convolutional neural network model

### Parameter extraction
* [`get_structure`](@ref) - get layer structure from a convolutional neural network model

### MIP formulation
* [`CNN_formulate!`](@ref) - formulate a `JuMP` model from the CNN

### Forward pass
* [`image_pass!`](@ref) - fix the input variables and optimize the model to get the ouput