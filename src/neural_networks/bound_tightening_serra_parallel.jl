using JuMP
using Gurobi

function copy_model(input_model)
    model = copy(input_model)
    set_solver_params!(model)
    return model
end

function set_solver_params!(model; threads=0, relax=false, time_limit=0)
    set_optimizer(model, () -> Gurobi.Optimizer(gurobi_env[myid()]))
    set_silent(model)
    #set_attribute(model, "Threads", 1)
    #relax_integrality(model)
    #set_attribute(model, "TimeLimit", 0.25)
end

function calculate_bounds(model::JuMP.Model, layer, neuron, W, b, neurons)

    @objective(model, Max, b[layer][neuron] + sum(W[layer][neuron, i] * model[:x][layer-1, i] for i in neurons(layer-1)))
    optimize!(model)
    @assert primal_status(model) == MOI.FEASIBLE_POINT "No solution found in time limit."
    
    ub_x = max(objective_bound(model), 0.0)

    set_objective_sense(model, MIN_SENSE)
    optimize!(model)
    @assert primal_status(model) == MOI.FEASIBLE_POINT "No solution found in time limit."
    ub_s = max(-objective_bound(model), 0.0)

    println("$neuron ")

    return ub_x, ub_s
end