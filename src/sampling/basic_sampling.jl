function optimize_by_sampling!(jump_model, samples)
    
    min = objective_sense(jump_model) == MIN_SENSE ? -1 : 1

    extremum = -Inf * min
    optimum = Vector{Float64}()

    for input in eachcol(samples)

        # run forward pass on a random sample
        forward_pass!(jump_model, input)

        # fix binary variables
        values = value.(jump_model[:z])
        [fix(jump_model[:z][key], values[key]) for key in keys(jump_model[:z].data)]

        # unfix the input after forward pass
        unfix.(jump_model[:x][0, :])

        # find local optimum (on hyperplane)
        optimize!(jump_model)

        x_opt = [value.(jump_model[:x][0, i]) for i in 1:length(jump_model[:x][0, :])]
        obj = objective_value(jump_model)

        if min * obj > extremum * min
            optimum = x_opt
            extremum = obj
        end

        # restore model
        unfix.(jump_model[:z])
        set_binary.(jump_model[:z])

    end

    return optimum, extremum

end