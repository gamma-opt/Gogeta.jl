# Tree ensemble models

This folder contains the Julia implementation of converting a tree ensmble model into a mixed-integer optimization model.

`tree_model_to_MIP.jl` - Contains the function for creating a JuMP model from a universal tree ensemble data type `TEModel`.

`types.jl` - Contains the definition for the universal datatype `TEModel` and the functions for converting different tree ensemble models into it.

`util.jl` - Contains auxiliary functions.