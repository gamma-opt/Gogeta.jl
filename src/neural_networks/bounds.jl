function copy_model(input_model, solver_params)
    model = copy(input_model)
    set_solver_params!(model, solver_params)
    return model
end

function set_solver_params!(model, params)
    if params.solver == "Gurobi"
        set_optimizer(model, () -> Gurobi.Optimizer(GUROBI_ENV[]))
        params.time_limit != 0 && set_attribute(model, "TimeLimit", params.time_limit)
        params.threads != 0 && set_attribute(model, "Threads", params.threads)
    elseif params.solver == "GLPK"
        set_optimizer(model, () -> GLPK.Optimizer())
        params.time_limit != 0 && set_attribute(model, "tm_lim", params.time_limit)
    else
        error("Solver has to be \"Gurobi\" or \"GLPK\"")
    end
    
    params.silent && set_silent(model)
    params.relax && relax_integrality(model)

end

function calculate_bounds(model::JuMP.Model, layer, neuron, W, b, neurons; layers_removed=0)

    @objective(model, Max, b[layer][neuron] + sum(W[layer][neuron, i] * model[:x][layer-1-layers_removed, i] for i in neurons(layer-1-layers_removed)))
    optimize!(model)
    
    upper_bound = if termination_status(model) == OPTIMAL
        objective_value(model)
    else
        @warn "Layer $layer, neuron $neuron could not be solved to optimality."
        objective_bound(model)
    end

    set_objective_sense(model, MIN_SENSE)
    optimize!(model)
 
    lower_bound = if termination_status(model) == OPTIMAL
        objective_value(model)
    else
        @warn "Layer $layer, neuron $neuron could not be solved to optimality."
        objective_bound(model)
    end

    println("Neuron: $neuron, bounds: [$lower_bound, $upper_bound]")

    return upper_bound, lower_bound
end