using JuMP
using Gurobi
using GLPK

const GUROBI_ENV = Ref{Gurobi.Env}()
global const BIG_M = Ref{Float64}(1.0e10)

function __init__()
    const GUROBI_ENV[] = Gurobi.Env()
end

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
        global BIG_M[] = 1_000.0 # GLPK cannot handle big-M constraints with large values
    else
        error("Solver has to be \"Gurobi\"/\"GLPK\"")
    end
    
    params.silent && set_silent(model)
    params.relax && relax_integrality(model)

end

function calculate_bounds(model::JuMP.Model, layer, neuron, W, b, neurons; layers_removed=0)

    @objective(model, Max, b[layer][neuron] + sum(W[layer][neuron, i] * model[:x][layer-1-layers_removed, i] for i in neurons(layer-1-layers_removed)))
    optimize!(model)
    
    upper_bound = if termination_status(model) == OPTIMAL
        max(objective_value(model), 0.0)
    else
        @warn "Layer $layer, neuron $neuron could not be solved to optimality."
        max(objective_bound(model), 0.0)
    end

    set_objective_sense(model, MIN_SENSE)
    optimize!(model)
 
    lower_bound = if termination_status(model) == OPTIMAL
        min(objective_value(model), 0.0)
    else
        @warn "Layer $layer, neuron $neuron could not be solved to optimality."
        min(objective_bound(model), 0.0)
    end

    println("Neuron: $neuron")

    return upper_bound, lower_bound
end

function calculate_bounds_fast(model::JuMP.Model, layer, neuron, W, b, neurons, layers_removed)

    upper = 1.0e10
    lower = -1.0e10

    function bounds_callback(cb_data, cb_where::Cint)

        # Only run at integer solutions
        if cb_where == GRB_CB_MIPSOL

            objbound = Ref{Cdouble}()
            objbest = Ref{Cdouble}()
            GRBcbget(cb_data, cb_where, GRB_CB_MIPSOL_OBJBND, objbound)
            GRBcbget(cb_data, cb_where, GRB_CB_MIPSOL_OBJBST, objbest)

            if objective_sense(model) == MAX_SENSE

                if objbest[] > 0
                    upper = min(objbound[], 1.0e10)
                    GRBterminate(backend(model))
                end
                
                if objbound[] <= 0
                    upper = max(objbound[], 0.0)
                    GRBterminate(backend(model))
                end

            elseif objective_sense(model) == MIN_SENSE

                if objbest[] < 0
                    lower = max(objbound[], -1.0e10)
                    GRBterminate(backend(model))
                end
                
                if objbound[] >= 0
                    lower = min(objbound[], 0.0)
                    GRBterminate(backend(model))
                end
            end
        end

    end

    @objective(model, Max, b[layer][neuron] + sum(W[layer][neuron, i] * model[:x][layer-1-layers_removed, i] for i in neurons(layer-1-layers_removed)))

    set_attribute(model, "LazyConstraints", 1)
    set_attribute(model, Gurobi.CallbackFunction(), bounds_callback)

    optimize!(model)

    set_objective_sense(model, MIN_SENSE)
    optimize!(model)

    status = if upper <= 0
        "stabily inactive"
    elseif lower >= 0
        "stabily active"
    else
        "normal"
    end
    println("Neuron: $neuron, $status, bounds: [$lower, $upper]")

    set_attribute(jump_model, Gurobi.CallbackFunction(), (cb_data, cb_where::Cint)->nothing)

    return upper, lower
end