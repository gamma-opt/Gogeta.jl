"""
    function optimize_by_walking!(original::JuMP.Model, nn_model::Flux.Chain, U_in, L_in; delta=0.1, return_sampled=false, logging=true, iterations=10, infeasible_per_iter=5)

Performs the full relaxing walk algorithm on the given neural network JuMP formulation. See Tong et al. (2024) for more details.

# Parameters
- `original`: `JuMP` model containing the NN formulation.
- `nn_model`: the original NN as `Flux.Chain`

# Optional Parameters
- `delta`: controls how strongly certain neurons are preferred when fixing the binary variables
- `return_sampled`: return sampled points in addition to the optima 
- `logging`: print progress info to the console
- `iterations`: the number of fresh starts from the linear relaxation (no binary variables fixed)
- `infeasible_per_iter`: the number of infeasible LP relaxations allowed before starting next iteration

"""
function optimize_by_walking!(original::JuMP.Model, nn_model::Flux.Chain, U_in, L_in; delta=0.1, return_sampled=false, logging=true, iterations=10, infeasible_per_iter=5)

    sample(items, weights) = items[findfirst(cumsum(weights) .> rand() * sum(weights))]
    
    opt_time = 0.0
    search_time = 0.0

    n_layers, _ = maximum(keys(original[:x].data))
    
    x_opt = Vector{Vector}() # list of locally optimum solutions
    opt = Vector{Float64}()
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
    
    local_opt, opt_value = local_search(x_tilde, original, nn_model, U_in, L_in)
    push!(x_opt, local_opt)
    push!(opt, opt_value)

    for iter in 1:iterations # TODO time limit based termination

        logging && println("\n\nIteration: $iter")
        infeasible_count = 0

        x_bar = deepcopy(x_tilde)
        z_bar = deepcopy(z_tilde)

        # determine active neurons
        binary_vars = keys(original[:z].data)
        values_flux = [map(n -> n > 0 ? 1.0 : 0.0, nn_model[1:layer](Float32.(x_bar))) for layer in unique(first.(binary_vars))]
        active = filter(key -> values_flux[key[1]][key[2]] == 1.0, binary_vars)

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

                @assert any(is_binary.(jump_model[:z])) == false || all(is_fixed.(jump_model[:z])) "Model must not have any integer variables."
                opt_time += @elapsed optimize!(jump_model)
                if termination_status(jump_model) == INFEASIBLE

                    unfix(jump_model[:z][layer, deleted])
                    set_binary(jump_model[:z][layer, deleted])
                    relax_integrality(jump_model)

                    logging && print("o")

                    infeasible_count += 1
                    if infeasible_count % infeasible_per_iter == 0
                        break
                    end
                else
                    x_bar = [value(jump_model[:x][0, i]) for i in 1:length(U_in)]
                    z_bar = value.(jump_model[:z])

                    logging && print(".")

                    if (x_bar in sampled_points) == false
                        search_time += @elapsed begin
                        local_opt, opt_value = local_search(x_bar, original, nn_model, U_in, L_in)
                        end
                        push!(x_opt, local_opt)
                        push!(opt, opt_value)
                        
                        push!(sampled_points, x_bar)
                    end
                end
            end

            if infeasible_count % infeasible_per_iter == 0
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
   
    end

    logging && println("\n\nLP OPTIMIZATION TIME: $opt_time")
    logging && println("LOCAL SEARCH TIME: $search_time")

    if return_sampled
        return x_opt, opt, sampled_points
    else
        return x_opt, opt
    end

end

"""
    function local_search(start, jump_model, nn_model, U_in, L_in; max_iter=100, epsilon=0.01, show_path=false, logging=false, tolerance=0.001)

Performs relaxing walk local search on the given neural network JuMP formulation. See Tong et al. (2024) for more details.

# Parameters
- `start`: starting point for the search (coordinate in the space)
- `jump_model`: `JuMP` model containing the NN formulation

# Optional Parameters
- `epsilon`: controls the step size taken out of the linear region
- `show_path`: return the path taken by the local search in addition to the optimum
- `logging`: print progress info to console
- `tolerance`: minimum relative improvement required at every step to continue the search

"""
function local_search(start, jump_model, U_in, L_in; max_iter=100, epsilon=0.01, show_path=false, logging=false, tolerance=0.001)

    x0 = deepcopy(start)
    x1 = deepcopy(x0)

    min = objective_sense(jump_model) == MIN_SENSE ? -1 : 1

    logging && println("\nStarting local search from: $start")

    path = Vector{Vector}()
    x0_obj = min * -Inf
    binary_vars = filter(is_binary, all_variables(jump_model))

    for iter in 1:max_iter

        logging && println("\nSEARCH STEP: $iter")

        push!(path, x0)
        
        # determine neuron activations and fix them
        [fix(jump_model[:x][0, i], x0[i]) for i in 1:length(start)]
        optimize!(jump_model)
        x0_obj = objective_value(jump_model)

        logging && println("Input objective: $x0_obj")

        binary_vals = map(value, binary_vars)
        foreach(pair -> fix(pair[1], pair[2]; force=true), zip(binary_vars, binary_vals))
        
        unfix.(jump_model[:x][0, :])

        # find hyperplane corner
        restore_integrality = relax_integrality(jump_model)
        optimize!(jump_model)

        x1 = [value(jump_model[:x][0, i]) for i in 1:length(start)]
        x1_obj = objective_value(jump_model)

        foreach(unfix, binary_vars)
        restore_integrality()

        logging && println("Corner objective: $x1_obj")
        
        if isapprox(x0_obj, x1_obj; rtol=tolerance) || min * x0_obj > x1_obj * min
            logging && println("Local optimum found: $x0")
            break
        end

        if min * x1_obj > x0_obj * min
            
            dir = x1 - x0
            x0 = x1 + epsilon*dir

            for i in 1:length(start)
                if (L_in[i] <= x0[i] <= U_in[i]) == false # outside domain
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