function load_drug_data(dataset_name)

    train_data = CSV.read(string(@__DIR__)*"/data/"*dataset_name*"_training_disguised.csv", DataFrame)
    train_names = names(train_data)[3:end]

    test_data = CSV.read(string(@__DIR__)*"/data/"*dataset_name*"_test_disguised.csv", DataFrame)
    test_names = names(test_data)[3:end]

    for col_name in setdiff(test_names, train_names)
        col_to_add = zeros(Int64, size(train_data)[1])
        train_data[!, col_name] = col_to_add
    end

    feat_names = sort(unique([train_names; test_names]))
    select!(train_data, ["MOLECULE"; "Act"; feat_names])

    for col_name in setdiff(train_names, test_names)
        col_to_add = zeros(Int64, size(test_data)[1])
        test_data[!, col_name] = col_to_add
    end

    select!(test_data, ["MOLECULE"; "Act"; feat_names])

    x_train = Matrix(train_data[:, 3:end]);
    y_train = Vector(train_data[:, 2]);

    x_test = Matrix(test_data[:, 3:end]);
    y_test = Vector(test_data[:, 2]);

    return train_data, x_train, y_train, x_test, y_test, feat_names
    
end

function load_concrete_data(dataset_name)

    data = CSV.read(string(@__DIR__)*"/data/"*dataset_name, DataFrame)
    feat_names = names(data)[1:8]

    Random.seed!(1)
    data = data[shuffle(1:end), :]

    train_split::Int = floor(0.75 * length(data[:, 1]));
    train_data = data[1:train_split, :];

    x_train = data[1:train_split, 1:8];
    y_train = data[1:train_split, 9];

    x_test = data[train_split+1:end, 1:8];
    y_test = data[train_split+1:end, 9];

    return train_data, x_train, y_train, x_test, y_test, feat_names

end

function train_evo_models(depths, trees, train_data, feat_names, x_train, y_train, x_test, y_test, filename, dataset_name, target)

    for depth in depths

        config = EvoTreeRegressor(nrounds=maximum(trees), max_depth=depth);
        train_time = @elapsed model = fit_evotree(config, train_data; target_name=target, verbosity=0, fnames=feat_names);

        result_file = open(string(@__DIR__)*"/test_results/"*filename, "a");
        write(result_file, "\nDataset: $dataset_name, Trees: $(maximum(trees)), Depth: $depth, Train time: $(train_time)\n");
        close(result_file)
    
        EvoTrees.save(model, string(@__DIR__)*"/trained_models/$(dataset_name)_$(maximum(trees))_trees_$(depth)_depth.bson")
    
        loaded_model = EvoTrees.load(string(@__DIR__)*"/trained_models/$(dataset_name)_$(maximum(trees))_trees_$(depth)_depth.bson");
    
        for forest_size in trees
    
            pred_train = EvoTrees.predict(loaded_model, x_train; ntree_limit=forest_size);
            pred_test = EvoTrees.predict(loaded_model, x_test; ntree_limit=forest_size);
    
            r2_score_train = 1 - sum((y_train .- pred_train).^2) / sum((y_train .- mean(y_train)).^2)
            r2_score_test = 1 - sum((y_test .- pred_test).^2) / sum((y_test .- mean(y_test)).^2)
    
            result_file = open(string(@__DIR__)*"/test_results/"*filename, "a");
            write(result_file, "Dataset: $dataset_name, Trees: $forest_size, Depth: $depth, R2 train: $(r2_score_train), R2 test: $(r2_score_test)\n");
            close(result_file)
    
        end
    
    end
end

function optimize_models(trees, depths, dataset_name, filename; time_limit=100)

    for depth in depths

        loaded_model = EvoTrees.load(string(@__DIR__)*"/trained_models/$(dataset_name)_$(maximum(trees))_trees_$(depth)_depth.bson");

        result_file = open(string(@__DIR__)*"/test_results/"*filename, "a");
        write(result_file, "\n");
        close(result_file)
        
        for forest_size in trees
    
            universal_model = extract_evotrees_info(loaded_model; tree_limit=forest_size+1);
    
            x_new, m_new, init_cons, n_creation_time, n_opt_time = tree_model_to_MIP(universal_model; create_initial=true, objective=MAX_SENSE, gurobi_env=ENV, timelimit=time_limit);
            x_alg, m_alg, gen_cons, alg_creation_time, alg_opt_time = tree_model_to_MIP(universal_model; create_initial=false, objective=MAX_SENSE, gurobi_env=ENV, timelimit=time_limit);
    
            result_file = open(string(@__DIR__)*"/test_results/"*filename, "a");
            normal_status = termination_status(m_new) == MOI.OPTIMAL ? "Optimality: $(objective_value(m_new))" : "Gap: $(relative_gap(m_new))"
            alg_status = termination_status(m_alg) == MOI.OPTIMAL ? "Optimality: $(objective_value(m_alg))" : "Gap: $(relative_gap(m_alg))"
            write(result_file, "Dataset: $dataset_name, Trees: $(forest_size), Depth: $depth, Normal time: $(n_creation_time) + $(n_opt_time) - $(normal_status), Alg time: $(alg_creation_time) + $(alg_opt_time) - $(alg_status), N levels: $(length(eachindex(m_new[:x]))), N leaves: $(length(eachindex(m_new[:y]))), Initial constraints: $(init_cons), Generated constraints: $(gen_cons)\n");
            close(result_file)
        end
    end
end