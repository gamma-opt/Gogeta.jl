function optimize_by_walking!(original::JuMP.Model, U_in, L_in; delta=0.01, return_sampled=false, logging=true, iterations=10, samples_per_iter=5)

    sample(items, weights) = items[findfirst(cumsum(weights) .> rand() * sum(weights))]
    opt_time = 0.0
    search_time = 0.0
    active_check_time = 0.0

    x_range = LinRange{Float32}(L_in[1], U_in[1], 20)
    y_range = LinRange{Float32}(L_in[2], U_in[2], 20)
    plots = Vector{Any}()

    n_layers, _ = maximum(keys(original[:x].data))
    
    x_opt = Vector{Vector}() # list of locally optimum solutions
    sampled_points = Vector{Vector}()

    # copy model
    jump_model = JuMP.copy(original)
    Main.set_solver!(jump_model)

    # get LP solution
    relax_integrality(jump_model)
    opt_time += @elapsed optimize!(jump_model)

    x_tilde = [value(jump_model[:x][0, i]) for i in 1:length(U_in)]
    z_tilde = value.(jump_model[:z])

    push!(sampled_points, x_tilde)
    push!(x_opt, local_search(x_tilde, original, U_in, L_in))

    for iter in 1:iterations # TODO time limit based termination

        logging && println("\n\nIteration: $iter")
        sample_count = 0

        x_bar = deepcopy(x_tilde)
        z_bar = deepcopy(z_tilde)

        # determine active neurons
        [fix(original[:x][0, i], x_bar[i]) for i in eachindex(x_bar)]
        active_check_time += @elapsed optimize!(original)

        z_values = value.(original[:z])
        binary_vars = collect(keys(original[:z].data))
        filter!(key -> z_values[key] == 1, binary_vars)
        active = binary_vars

        unfix.(original[:x][0, :])

        for layer in 1:(n_layers-1) # hidden layers

            logging && println("\n--------------Layer: $layer--------------")

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
                    fix(jump_model[:z][layer, deleted], 0.0; force=true)
                else
                    fix(jump_model[:z][layer, deleted], 1.0; force=true)
                end

                # push!(plots, 
                #     plot(x_range, y_range, (x, y) -> forward_pass!(jump_model, [x, y])[], st=:contourf, c=cgrad(:viridis, scale=:log), lw=0, colorbar=false, axis=false, size=(300, 300), framestyle=:none, margin = 0*Plots.mm)
                # )
                # unfix.(jump_model[:x][0, :])

                @assert any(is_binary.(jump_model[:z])) == false || all(is_fixed.(jump_model[:z])) "Model must not have any integer variables."
                opt_time += @elapsed optimize!(jump_model)
                if termination_status(jump_model) == INFEASIBLE

                    unfix(jump_model[:z][layer, deleted])
                    set_binary(jump_model[:z][layer, deleted])
                    relax_integrality(jump_model)

                    logging && print("o")
                else
                    x_bar = [value(jump_model[:x][0, i]) for i in 1:length(U_in)]
                    z_bar = value.(jump_model[:z])

                    logging && print(".")

                    if (x_bar in sampled_points) == false
                        search_time += @elapsed push!(x_opt, local_search(x_bar, original, U_in, L_in))
                        push!(sampled_points, x_bar)

                        sample_count += 1
                        if sample_count % samples_per_iter == 0
                            break
                        end
                    end
                end
            end

            if sample_count % samples_per_iter == 0
                break
            end

        end # start new iteration

        # reset binary variables - start all over
        for var in jump_model[:z]
            if is_fixed(var)
                unfix(var)
            end
        end

        set_binary.(jump_model[:z])
        relax_integrality(jump_model)

        # lrs = plot(plots..., layout=(20, 10), size=(300*10, 300*20))
        # savefig(lrs, "spaces.png")

    end

    println("\n\nLP OPTIMIZATION TIME: $opt_time")
    println("LOCAL SEARCH TIME: $search_time")
    println("ACTIVE NEURONS CHECKING TIME: $active_check_time")

    if return_sampled
        return x_opt, sampled_points
    else
        return x_opt
    end

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
        @assert any(is_binary.(jump_model[:z])) == false || all(is_fixed.(jump_model[:z])) "Model must not have any integer variables."
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