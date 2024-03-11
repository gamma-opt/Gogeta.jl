function optimize_by_walking!(original::JuMP.Model, U_in, L_in; delta=0.0)

    sample(items, weights) = items[findfirst(cumsum(weights) .> rand() * sum(weights))]

    n_layers, _ = maximum(keys(original[:x].data))
    
    x_opt = Vector{Vector}() # list of locally optimum solutions

    # copy model
    jump_model = JuMP.copy(original)
    Main.set_solver!(jump_model)

    # get LP solution
    relax_integrality(jump_model)
    optimize!(jump_model)

    x_tilde = [value(jump_model[:x][0, i]) for i in 1:length(U_in)]
    z_tilde = value.(jump_model[:z])
    
    push!(x_opt, local_search(x_tilde, original, U_in, L_in))

    for iter in 1:5 # TODO time limit based termination

        println("\nIteration: $iter")

        x_bar = deepcopy(x_tilde)
        z_bar = deepcopy(z_tilde)

        # determine active neurons
        [fix(original[:x][0, i], x_bar[i]) for i in eachindex(x_bar)]
        optimize!(original)

        z_values = value.(original[:z])
        binary_vars = collect(keys(original[:z].data))
        filter!(key -> z_values[key] == 1, binary_vars)
        active = binary_vars

        unfix.(original[:x][0, :])

        for layer in 1:(n_layers-1) # hidden layers

            println("\n--------------Layer: $layer--------------")

            n_neurons = first(maximum(keys(original[:x][layer, :].data)))
            neurons = Set(1:n_neurons)

            while isempty(neurons) == false

                ksi = Dict{Int, Float64}()
                for neuron in neurons
                    if (layer, neuron) in active
                        ksi[neuron] = 1 - z_bar[layer, neuron]
                    else
                        ksi[neuron] = z_bar[layer, neuron]
                    end
                end

                neurons_ordered = sort(collect(neurons))
                ksi_ordered = last.(sort(collect(ksi)))
                total_prob = sum(Base.values(ksi)) + delta * length(ksi)

                weights = ksi_ordered .+ (delta / total_prob)

                deleted = try # something wrong with this
                    sample(neurons_ordered, weights)
                catch e
                    rand(neurons_ordered)
                end
                delete!(neurons, deleted)

                if (layer, deleted) in active
                    fix(jump_model[:z][layer, deleted], 0.0; force=true)
                    println("Fixed z[$layer, $deleted] to 0.0")
                else
                    fix(jump_model[:z][layer, deleted], 1.0; force=true)
                    println("Fixed z[$layer, $deleted] to 1.0")
                end

                optimize!(jump_model)
                if termination_status(jump_model) != INFEASIBLE
                    x_bar = [value(jump_model[:x][0, i]) for i in 1:length(U_in)]
                    z_bar = value.(jump_model[:z])

                    push!(x_opt, local_search(x_bar, original, U_in, L_in))
                else
                    unfix(jump_model[:z][layer, deleted])
                end
            end
        end

        # reset binary variables - start all over
        for var in jump_model[:z]
            if is_fixed(var)
                unfix(var)
            end
        end

        set_binary.(jump_model[:z])
        relax_integrality(jump_model)

    end

    return x_opt

end

function local_search(start, jump_model, U_in, L_in; epsilon=0.01, show_path=false, logging=false)

    x0 = deepcopy(start)
    x1 = deepcopy(x0)

    logging && println("\nStarting local search from: $start")

    path = Vector{Vector}()

    binary_vars = keys(jump_model[:z].data)

    while true

        push!(path, x0)

        [fix(jump_model[:x][0, i], x0[i]) for i in eachindex(start)]
        optimize!(jump_model)
        x0_obj = objective_value(jump_model)

        # fix binary variables
        values = value.(jump_model[:z])
        [fix(jump_model[:z][key], values[key]) for key in binary_vars]

        # unfix input after forward pass
        unfix.(jump_model[:x][0, :])

        # find hyperplane corner
        optimize!(jump_model)

        x1 = [value.(jump_model[:x][0, i]) for i in 1:length(start)]
        x1_obj = objective_value(jump_model)
        push!(path, x1)

        unfix.(jump_model[:z])

        if x1_obj > x0_obj
            
            dir = x1 - x0
            x0 = x1 + epsilon*dir

            for i in 1:length(start)
                if (L_in[i] <= x0[i] <= U_in[i]) == false # outside domain
                    x0[i] = x1[i]
                end
            end

            # check if next point is better
            [fix(jump_model[:x][0, i], x0[i]) for i in eachindex(start)]
            optimize!(jump_model)
            x0_obj = objective_value(jump_model)

            # unfix the input after forward pass
            unfix.(jump_model[:x][0, :])
            
        end

        if isapprox(x0_obj, x1_obj; atol=1e-5) || x0_obj < x1_obj # TODO what should the tolerance be
            break 
        end
    end

    logging && println("Local optimum found: $x1")
    if show_path 
        return x1, path
    else 
        return x1
    end

end