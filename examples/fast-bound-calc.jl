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