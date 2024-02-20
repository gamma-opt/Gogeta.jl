"""
    SolverParams

Parameters to be used by the solver.

# Fields
- `solver`: has to be "Gurobi" or "GLPK"
- `silent`: is the solver log shown
- `threads`: use 0 for solver default
- `relax`: linear relaxation for the MIP
- `time_limit`: time limit for each optimization in the model

# Examples
```julia
julia> solver_params = SolverParams(solver="Gurobi", silent=true, threads=0, relax=false, time_limit=0);
```
"""
@kwdef struct SolverParams
    solver::String
    silent::Bool
    threads::Int
    relax::Bool
    time_limit::Float64
end

"""
    function forward_pass!(jump_model::JuMP.Model, input)

Calculates the output of a neural network -representing JuMP model given some input.
"""
function forward_pass!(jump_model::JuMP.Model, input)
    
    try
        @assert length(input) == length(jump_model[:x][0, :]) "Incorrect input length."
        [fix(jump_model[:x][0, i], input[i], force=true) for i in eachindex(input)]
        optimize!(jump_model)
        (last_layer, outputs) = maximum(keys(jump_model[:x].data))
        result = value.(jump_model[:x][last_layer, :])
        return [result[i] for i in 1:outputs]
    catch e
        println("Input outside of input bounds or incorrectly constructed model.")
        return [nothing]
    end

end

"""
    function NN_to_MIP_with_precomputed(NN_model::Flux.Chain, U_in, L_in, solver_params, U_bounds, L_bounds; silent=false)

Creates a mixed-integer optimization problem from a `Flux.Chain` model. In this version, the neuron activation bounds must be given as arguments.

Returns the resulting JuMP model.

# Parameters
- `NN_model`: the neural network as a `Flux.Chain`
- `U_in`: upper bounds of the input neurons
- `L_in`: lower bounds of the input neurons
- `solver_params`: a `SolverParams` struct
- `U_bounds`: precomputed upper activation bounds for each layer
- `L_bounds`: precomputed lower activation bounds for each layer

# Optional arguments
- `silent`: controls the output logs

"""
function NN_to_MIP_with_precomputed(NN_model::Flux.Chain, U_in, L_in, solver_params, U_bounds, L_bounds; silent=false)
    return formulate_and_compress(NN_model, U_in, L_in; U_bounds, L_bounds, solver_params, silent)
end

"""
    function NN_to_MIP_with_bound_tightening(NN_model::Flux.Chain, U_in, L_in, solver_params; bound_tightening="fast", U_out=nothing, L_out=nothing, silent=false)

Creates a mixed-integer optimization problem from a `Flux.Chain` model. In this version, the neuron activation bounds are calculated as the model is being created.

Returns the resulting JuMP model and the computed activation bounds.

# Parameters
- `NN_model`: the neural network as a `Flux.Chain`
- `U_in`: upper bounds of the input neurons
- `L_in`: lower bounds of the input neurons
- `solver_params`: a `SolverParams` struct

# Optional arguments
- `bound_tightening`: "fast", "standard" or "output"
- `silent`: controls the output logs
- `U_out`: upper activation bounds for the output layer (must be set if bound_tightening="output")
- `L_out`: lower activation bounds for the output layer (must be set if bound_tightening="output")

"""
function NN_to_MIP_with_bound_tightening(NN_model::Flux.Chain, U_in, L_in, solver_params; bound_tightening="fast", U_out=nothing, L_out=nothing, silent=false)

    if bound_tightening == "output" @assert U_out !== nothing && L_out !== nothing "Provide output bounds." end

    return formulate_and_compress(NN_model, U_in, L_in; bound_tightening, solver_params, U_out, L_out, silent)
end

"""
    function compress_with_precomputed(NN_model::Flux.Chain, U_in, L_in, U_bounds, L_bounds; silent=true)

Creates a compressed version of the neural network model `Flux.Chain` given as input. This is accomplished by identifying inactive and stabily active neurons.

Return the compressed model and the list of removed neurons.

# Parameters
- `NN_model`: the neural network as a `Flux.Chain`
- `U_in`: upper bounds of the input neurons
- `L_in`: lower bounds of the input neurons
- `U_bounds`: precomputed upper activation bounds for each layer
- `L_bounds`: precomputed lower activation bounds for each layer

# Optional arguments
- `silent`: controls the output logs

"""
function compress_with_precomputed(NN_model::Flux.Chain, U_in, L_in, U_bounds, L_bounds; silent=true)
    return formulate_and_compress(NN_model, U_in, L_in; U_bounds, L_bounds, compress=true, silent)
end

"""
    function compress_with_bound_tightening(NN_model::Flux.Chain, U_in, L_in, solver_params; bound_tightening="fast", silent=false)

Creates a compressed version of the neural network model `Flux.Chain` given as input. This is accomplished by identifying inactive and stabily active neurons.
This version computes the bounds as the network is being compressed.

Return the compressed model, the list of removed neurons, the JuMP model and computed bounds.

# Parameters
- `NN_model`: the neural network as a `Flux.Chain`
- `U_in`: upper bounds of the input neurons
- `L_in`: lower bounds of the input neurons
- `solver_params`: a `SolverParams` struct

# Optional arguments
- `bound_tightening`: "fast" or "standard"
- `silent`: controls the output logs

"""
function compress_with_bound_tightening(NN_model::Flux.Chain, U_in, L_in, solver_params; bound_tightening="fast", silent=false)
    return formulate_and_compress(NN_model, U_in, L_in; solver_params, bound_tightening, compress=true, silent)
end