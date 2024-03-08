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

    for _ in 1:5 # TODO time limit based termination

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

                deleted = sample(neurons_ordered, weights)
                delete!(neurons, deleted)

                if (layer, deleted) in active
                    fix(jump_model[:z][layer, deleted], 0.0)
                else
                    fix(jump_model[:z][layer, deleted], 1.0)
                end

                optimize!(jump_model)
                if termination_status(jump_model) == FEASIBLE_POINT
                    x_bar = [value(jump_model[:x][0, i]) for i in 1:length(U_in)]
                    z_bar = value.(jump_model[:z])

                    push!(x_opt, local_search(x_bar, original, U_in, L_in))
                else
                    unfix(jump_model[:z][layer, deleted])
                end
            end
        end

        unfix.(jump_model[:z])

    end

    return x_opt

end

function local_search(start, jump_model, U_in, L_in; epsilon=0.1)

    x0 = deepcopy(start)
    x1 = deepcopy(x0)

    path = Vector{Vector}()

    binary_vars = keys(jump_model[:z].data)

    while true

        push!(path, x0)

        [fix(jump_model[:x][0, i], x0[i]) for i in eachindex(start)]
        optimize!(jump_model)
        init_obj = objective_value(jump_model)

        println("\nCurrently at: $x0 with value $init_obj")

        # fix binary variables
        values = value.(jump_model[:z])
        [fix(jump_model[:z][key], values[key]) for key in binary_vars]

        # unfix input after forward pass
        unfix.(jump_model[:x][0, :])

        # find hyperplane corner
        optimize!(jump_model)

        x1 = [value.(jump_model[:x][0, i]) for i in 1:length(start)]
        corner_obj = objective_value(jump_model)
        push!(path, x1)

        println("Corner at: $x1 with value $corner_obj")

        unfix.(jump_model[:z])

        if corner_obj > init_obj
            
            dir = x1 - x0
            x0 = x1 + epsilon*dir

            for i in 1:length(start)
                if (L_in[i] <= x0[i] <= U_in[i]) == false # outside domain
                    x0[i] = x1[i]
                end
            end
        end

        # check if next point is better
        [fix(jump_model[:x][0, i], x0[i]) for i in eachindex(start)]
        optimize!(jump_model)
        new_obj = objective_value(jump_model)
        println("New point at: $x0 with value $new_obj")

        # unfix the input after forward pass
        unfix.(jump_model[:x][0, :])

        if new_obj <= (1.0 + 1e-5) * corner_obj 
            break 
        end
    end

    return x1, path

end