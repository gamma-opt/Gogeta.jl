using JuMP
using Gurobi
using GLPK

const GUROBI_ENV = Ref{Gurobi.Env}()

function __init__()
    const GUROBI_ENV[] = Gurobi.Env()
end

function copy_model(input_model, solver_params)
    model = copy(input_model)
    set_solver_params!(model, solver_params)
    return model
end

function set_solver_params!(model, params)
    set_optimizer(model, () -> Gurobi.Optimizer(GUROBI_ENV[]))
    # set_optimizer(model, () -> GLPK.Optimizer())
    params.silent && set_silent(model)
    params.threads != 0 && set_attribute(model, "Threads", params.threads)
    params.relax && relax_integrality(model)
    params.time_limit != 0 && set_attribute(model, "TimeLimit", params.time_limit)
    # params.time_limit != 0 && set_attribute(model, "tm_lim", params.time_limit)
end

function calculate_bounds(model::JuMP.Model, layer, neuron, W, b, neurons)

    @objective(model, Max, b[layer][neuron] + sum(W[layer][neuron, i] * model[:x][layer-1, i] for i in neurons(layer-1)))
    optimize!(model)
    
    upper_bound = if termination_status(model) == OPTIMAL
        max(objective_value(model), 0.0)
    else
        max(objective_bound(model), 0.0)
    end

    set_objective_sense(model, MIN_SENSE)
    optimize!(model)
 
    lower_bound = if termination_status(model) == OPTIMAL
        max(-objective_value(model), 0.0)
    else
        max(-objective_bound(model), 0.0)
    end

    println("Neuron: $neuron")

    return upper_bound, lower_bound
end

function calculate_bounds_fast(model::JuMP.Model, layer, neuron, W, b, neurons, layer_removed)

    function bounds_callback(cb_data, cb_where::Cint)

        # Only run at integer solutions
        if cb_where == GRB_CB_MIPSOL

            objbound = Ref{Cdouble}()
            objval = Ref{Cdouble}()
            GRBcbget(cb_data, cb_where, GRB_CB_MIPSOL_OBJBND, objbound)
            GRBcbget(cb_data, cb_where, GRB_CB_MIPSOL_OBJ, objval)

            if objective_sense(model) == MAX_SENSE

                if objval[] > 0
                    GRBterminate(backend(model))
                end

                if objbound[] <= 0
                    GRBterminate(backend(model))
                end

            elseif objective_sense(model) == MIN_SENSE

                if objval[] < 0
                    GRBterminate(backend(model))
                end
    
                if objbound[] >= 0
                    GRBterminate(backend(model))
                end
            end
        end

    end

    @objective(model, Max, b[layer][neuron] + sum(W[layer][neuron, i] * model[:x][layer-1-layer_removed, i] for i in neurons(layer-1-layer_removed)))

    set_attribute(model, "LazyConstraints", 1)
    set_attribute(model, Gurobi.CallbackFunction(), bounds_callback)

    optimize!(model)
    upper = objective_bound(model)

    set_objective_sense(model, MIN_SENSE)
    optimize!(model)
    lower = objective_bound(model)

    status = if upper <= 0
        "stabily inactive"
    elseif lower >= 0
        "stabily active"
    else
        "normal"
    end
    println("Neuron: $neuron, $status")

    return upper, lower
end