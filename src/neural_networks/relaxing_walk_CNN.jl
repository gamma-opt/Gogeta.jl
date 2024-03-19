function optimize_by_walking_CNN!(cnn_jump::JuMP.Model, input; iterations=10, samples_per_iter=5, timelimit=0.5)

    foreach(unfix, filter(is_fixed, all_variables(cnn_jump)))

    copied_model = copy(cnn_jump)
    set_solver!(copied_model)

    opt_time = 0.0
    
    binary_vars = findall(var -> is_binary(var) && name(var) != "", all_variables(copied_model))

    relax_integrality(copied_model)
    opt_time += @elapsed optimize!(copied_model)
    lp_sol = [value.(copied_model[:c][0, row, col, channel]) for row in eachindex(input[:, 1, 1, 1]), col in eachindex(input[1, :, 1, 1]), channel in eachindex(input[1, 1, :, 1])]

    samples = Vector{Array{Float64, 3}}()
    for iter in 1:iterations

        [fix(cnn_jump[:c][0, row, col, channel], lp_sol[row, col, channel, 1], force=true) for row in eachindex(lp_sol[:, 1, 1, 1]), col in eachindex(lp_sol[1, :, 1, 1]), channel in eachindex(lp_sol[1, 1, :, 1])]
        opt_time += @elapsed optimize!(cnn_jump)

        active = Set(first.(filter(
            pair -> pair[2] == 1.0,
            map(i -> i => value(all_variables(cnn_jump)[i]), binary_vars)
        )))
        foreach(unfix, filter(is_fixed, all_variables(cnn_jump))) # unfix input
        
        println("\nITERATION $iter")
        
        for s in 1:samples_per_iter
            
            to_be_fixed = rand(binary_vars)
            while is_fixed(all_variables(copied_model)[to_be_fixed])
                to_be_fixed = rand(binary_vars)
            end

            if to_be_fixed in active
                println("\nFixed $(name(all_variables(copied_model)[to_be_fixed])) to 0")
                fix(all_variables(copied_model)[to_be_fixed], 0.0; force=true)
            else
                println("\nFixed $(name(all_variables(copied_model)[to_be_fixed])) to 1")
                fix(all_variables(copied_model)[to_be_fixed], 1.0; force=true)
            end
            
            # re-optimize
            set_attribute(copied_model, "TimeLimit", timelimit)
            opt_time += @elapsed optimize!(copied_model)
            
            if termination_status(copied_model) != OPTIMAL
                println("Infeasible or timeout")
                unfix(all_variables(copied_model)[to_be_fixed])
                set_binary(all_variables(copied_model)[to_be_fixed])
                relax_integrality(copied_model)
                break
            else
                lp_sol = [value.(copied_model[:c][0, row, col, channel]) for row in eachindex(input[:, 1, 1, 1]), col in eachindex(input[1, :, 1, 1]), channel in eachindex(input[1, 1, :, 1])]

                if all(map(sample -> norm(lp_sol-sample, 2) / norm(lp_sol, 2) >= 0.05, samples))

                    push!(samples, lp_sol)

                    if iszero(lp_sol)
                        println("Zero solution")
                    else
                        println("Objective value: $(CNN_model(reshape(lp_sol, 70, 50, 1, 1))[1])")
                        display(heatmap(lp_sol[:, :, 1], background=false, legend=false, color=:inferno, aspect_ratio=:equal, axis=([], false)))
                        
                        # println("Starting local search...")
                        # x_opt, val_opt = local_search_CNN(lp_sol, cnn_jump)

                        # display(heatmap(x_opt[:, :, 1], background=false, legend=false, color=:inferno, aspect_ratio=:equal, axis=([], false)))
                    end
                else
                    println("Already sampled.")
                    break
                end
            end
        end

        set_attribute(copied_model, "TimeLimit", Inf)

        foreach(unfix, filter(is_fixed, all_variables(copied_model)))
        foreach(set_binary, all_variables(copied_model)[binary_vars])
        relax_integrality(copied_model)

    end

    println("\nSampling complete.\n")
    println("Samples: $(length(samples))")

    println("Optimization time: $opt_time")
end

function local_search_CNN(start, cnn_jump; epsilon=0.01, max_iter=10, show_path=false)

    nrows, ncols, nchannels, _ = size(start)

    x0 = deepcopy(start)
    x1 = deepcopy(x0)

    path = Vector{typeof(x0)}()
    
    min = objective_sense(cnn_jump) == MIN_SENSE ? -1 : 1
    x0_obj = min * Inf

    for iter in 1:max_iter

        println("\nSEARCH STEP: $iter")

        # display(heatmap(x0[:, :, 1], background=false, legend=false, title="input", color=:inferno, aspect_ratio=:equal, axis=([], false)))

        x0 = map(pixel -> if pixel > 1.0 1.0 elseif pixel < 0.0 0.0 else pixel end, x0)

        push!(path, x0)
        [fix(cnn_jump[:c][0, row, col, channel], x0[row, col, channel, 1], force=true) for row in 1:nrows, col in 1:ncols, channel in 1:nchannels];
        optimize!(cnn_jump)
        x0_obj = objective_value(cnn_jump)
        
        println("Input objective: $x0_obj")

        binary_vars = filter(is_binary, all_variables(cnn_jump))
        binary_vals = map(value, binary_vars)
        foreach(pair -> fix(pair[1], pair[2]; force=true), zip(binary_vars, binary_vals))

        unfix.(cnn_jump[:c][0, :, :, :])

        # find hyperplane corner
        restore_integrality = relax_integrality(cnn_jump)
        optimize!(cnn_jump)
        
        x1 = [value(cnn_jump[:c][0, row, col, channel]) for row in 1:nrows, col in 1:ncols, channel in 1:nchannels];
        x1_obj = objective_value(cnn_jump)        

        foreach(unfix, binary_vars)
        restore_integrality()

        println("Corner objective: $x1_obj")

        if isapprox(x1_obj, x0_obj; rtol=0.02)
            println("Not enough improvement to input. Terminating...")
            break
        end

        if min * x1_obj > x0_obj * min
            
            dir = x1 - x0
            x0 = x1 + epsilon*dir

            for i in eachindex(x0)
                if (0.0 <= x0[i] <= 1.0) == false # outside domain
                    x0[i] = x1[i]
                end
            end
            
        end
    end

    if show_path
        return x0, x0_obj, path
    else
        return x0, x0_obj
    end
end