using JuMP
using Gurobi

function copy_model(input_model)
    model = copy(input_model)
    set_solver_params!(model)
    return model
end

function set_solver_params!(model)
    set_optimizer(model, () -> Gurobi.Optimizer(GUROBI_ENV[myid()]))
    SILENT && set_silent(model)
    THREADS != 0 && set_attribute(model, "Threads", THREADS)
    RELAX && relax_integrality(model)
    LIMIT != 0 && set_attribute(model, "TimeLimit", LIMIT)
end

function calculate_bounds(model::JuMP.Model, layer, neuron, W, b, neurons)

    @objective(model, Max, b[layer][neuron] + sum(W[layer][neuron, i] * model[:x][layer-1, i] for i in neurons(layer-1)))
    optimize!(model)
    @assert primal_status(model) == MOI.FEASIBLE_POINT "No solution found in time limit."
    
    upper_bound = max(objective_bound(model), 0.0)

    set_objective_sense(model, MIN_SENSE)
    optimize!(model)
    @assert primal_status(model) == MOI.FEASIBLE_POINT "No solution found in time limit."
    lower_bound = max(-objective_bound(model), 0.0)

    println("Neuron: $neuron")

    return upper_bound, lower_bound
end