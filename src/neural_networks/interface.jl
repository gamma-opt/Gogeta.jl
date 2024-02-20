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

function NN_to_MIP_with_precomputed(NN_model::Flux.Chain, U_in, L_in, solver_params, U_bounds, L_bounds; silent=false)
    return formulate_and_compress(NN_model, U_in, L_in; U_bounds, L_bounds, solver_params, silent)
end

function NN_to_MIP_with_bound_tightening(NN_model::Flux.Chain, U_in, L_in, solver_params; bound_tightening="fast", U_out=nothing, L_out=nothing, silent=false)

    if bound_tightening == "output" @assert U_out !== nothing && L_out !== nothing "Provide output bounds." end

    return formulate_and_compress(NN_model, U_in, L_in; bound_tightening, solver_params, U_out, L_out, silent)
end

function compress_with_precomputed(NN_model::Flux.Chain, U_in, L_in, U_bounds, L_bounds; silent=true)
    return formulate_and_compress(NN_model, U_in, L_in; U_bounds, L_bounds, compress=true, silent)
end

function compress_with_bound_tightening(NN_model::Flux.Chain, U_in, L_in, solver_params; bound_tightening="fast", silent=false)
    return formulate_and_compress(NN_model, U_in, L_in; solver_params, bound_tightening, compress=true, silent)
end