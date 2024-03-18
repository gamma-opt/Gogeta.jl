function optimize_by_walking_CNN!(cnn_jump::JuMP.Model, input; iterations=100)

    foreach(unfix, filter(is_fixed, all_variables(cnn_jump)))

    copied_model = copy(cnn_jump)
    set_solver!(copied_model)

    binary_vars = filter(var -> is_binary(var) && name(var) != "", all_variables(copied_model))

    relax_integrality(copied_model)
    optimize!(copied_model)

    binary_set = collect(zip(binary_vars, map(value, binary_vars)))
    
    for iter in 1:iterations
        
        println("\nITERATION $iter")
        infeasible = 0
        
        while infeasible <= 5
            
            to_be_fixed = rand(filter(var_val -> is_fixed(var_val[1]) == false, binary_set))
            println("FIXED: $(name(to_be_fixed[1]))")
            fix(to_be_fixed[1], 1.0; force=true)
            
            # re-optimize
            optimize!(copied_model)
            binary_set = collect(zip(binary_vars, map(value, binary_vars)))
            
            if termination_status(copied_model) == INFEASIBLE
                println("Infeasible")
                unfix(to_be_fixed[1])
                set_binary(to_be_fixed[1])
                relax_integrality(copied_model)
                infeasible += 1
            else
                lp_sol = [value.(copied_model[:c][0, row, col, channel]) for row in eachindex(input[:, 1, 1, 1]), col in eachindex(input[1, :, 1, 1]), channel in eachindex(input[1, 1, :, 1])];

                if iszero(lp_sol)
                    println("Zero solution")
                    infeasible += 5
                else
                    display(heatmap(lp_sol[:, :, 1], background=false, legend=false, color=:inferno, aspect_ratio=:equal, axis=([], false)))
                    
                    println("Starting local search...")
                    x_opt, val_opt = local_search_CNN(lp_sol, cnn_jump)
                    display(heatmap(x_opt[:, :, 1], background=false, legend=false, color=:inferno, aspect_ratio=:equal, axis=([], false)))
                end
            end
        end

        foreach(unfix, filter(is_fixed, binary_vars))
        foreach(set_binary, binary_vars)
        relax_integrality(copied_model)

    end
end

function local_search_CNN(start, cnn_jump; epsilon=0.01, max_iter=10, show_path=false)

    x0 = deepcopy(start)
    x1 = deepcopy(x0)

    x0_obj = 0.0
    path = Vector{typeof(x0)}()

    min = objective_sense(cnn_jump) == MIN_SENSE ? -1 : 1

    for iter in 1:max_iter

        println("\nSEARCH STEP: $iter")

        # display(heatmap(x0[:, :, 1], background=false, legend=false, title="input", color=:inferno, aspect_ratio=:equal, axis=([], false)))

        # x0 = map(pixel -> if pixel > 1.0 1.0 elseif pixel < 0.0 0.0 else pixel end, x0)

        push!(path, x0)
        [fix(cnn_jump[:c][0, row, col, channel], x0[row, col, channel, 1], force=true) for row in eachindex(x0[:, 1, 1, 1]), col in eachindex(x0[1, :, 1, 1]), channel in eachindex(x0[1, 1, :, 1])];
        optimize!(cnn_jump)
        x0_obj = objective_value(cnn_jump)
        
        println("INPUT OBJECTIVE: $x0_obj")

        binary_vars = filter(is_binary, all_variables(cnn_jump))
        binary_vals = map(value, binary_vars)
        foreach(pair -> fix(pair[1], pair[2]; force=true), zip(binary_vars, binary_vals))

        unfix.(cnn_jump[:c][0, :, :, :]);

        # find hyperplane corner
        restore_integrality = relax_integrality(cnn_jump)
        optimize!(cnn_jump)
        
        x1 = [value.(cnn_jump[:c][0, row, col, channel]) for row in eachindex(x0[:, 1, 1, 1]), col in eachindex(x0[1, :, 1, 1]), channel in eachindex(x0[1, 1, :, 1])]
        x1_obj = objective_value(cnn_jump)
        
        restore_integrality()

        println("CORNER OBJECTIVE: $x1_obj")

        foreach(unfix, binary_vars)

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