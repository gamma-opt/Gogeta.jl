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

function train_evo_models(depths, trees, train_data, feat_names, x_train, y_train, x_test, y_test)

    for depth in depths

        config = EvoTreeRegressor(nrounds=maximum(trees), max_depth=depth);
        train_time = @elapsed model = fit_evotree(config, train_data[:, 2:end]; target_name="Act", verbosity=0, fnames=feat_names);
        
        result_file = open(string(@__DIR__)*"/drug_test_results.txt", "a");
        write(result_file, "\nDataset: $dataset_name, Trees: $(maximum(trees)), Depth: $depth, Train time: $(train_time)\n");
        close(result_file)
    
        EvoTrees.save(model, string(@__DIR__)*"/trained_models/$(dataset_name)_$(maximum(trees))_trees_$(depth)_depth.bson")
    
        loaded_model = EvoTrees.load(string(@__DIR__)*"/trained_models/$(dataset_name)_$(maximum(trees))_trees_$(depth)_depth.bson");
    
        for forest_size in trees
    
            pred_train = EvoTrees.predict(loaded_model, x_train; ntree_limit=forest_size);
            pred_test = EvoTrees.predict(loaded_model, x_test; ntree_limit=forest_size);
    
            r2_score_train = 1 - sum((y_train .- pred_train).^2) / sum((y_train .- mean(y_train)).^2)
            r2_score_test = 1 - sum((y_test .- pred_test).^2) / sum((y_test .- mean(y_test)).^2)
    
            result_file = open(string(@__DIR__)*"/drug_test_results.txt", "a");
            write(result_file, "Dataset: $dataset_name, Trees: $forest_size, Depth: $depth, R2 train: $(r2_score_train), R2 test: $(r2_score_test)\n");
            close(result_file)
    
        end
    
    end
end