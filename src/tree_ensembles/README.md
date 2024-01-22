# Tree ensemble models

This folder contains the Julia implementation of converting a tree ensmble model into a mixed-integer optimization model.

`tree_model_to_MIP.jl` - Contains the function for creating a JuMP model from a universal tree ensemble data type `TEModel` and the functions required to optimize the model - with or without lazy constraints.

`types.jl` - Contains the definition for the universal datatype `TEModel` and the functions for converting different tree ensemble models into it as well as the function for precalculating the child leaves.

`util.jl` - Contains auxiliary functions for calculating the children in a binary tree, and obtaining a numerical solution from the JuMP model.