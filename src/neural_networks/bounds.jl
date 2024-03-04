"""
    function copy_model(input_model, solver_params)

Creates a copy of a JuMP model. Solver has to be specified for each new copy. Used for parallelization.
"""
function copy_model(input_model)
    model = copy(input_model)
    try
        Main.set_solver!(model)
    catch e
        error("To use multiprocessing, 'set_solver!'-function must be defined in the global scope for each worker process.")
    end
    return model
end

"""
    function calculate_bounds(model::JuMP.Model, layer, neuron, W, b, neurons; layers_removed=0)

Calculates the upper and lower activation bounds for a neuron in a ReLU-activated neural network.
"""
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