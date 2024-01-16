"DATA LOADING"

dataset_name = "3A4";
train_data, x_train, y_train, x_test, y_test, feat_names = load_drug_data(dataset_name);

"TREE MODEL GENERATION"

trees = [10, 50, 100, 200, 350, 500, 750, 1000];
depths = [3, 5, 7, 9, 12];

depths = [9, 12];

train_evo_models(depths, trees, train_data[:, 2:end], feat_names, x_train, y_train, x_test, y_test, "drug_test_results.txt", dataset_name, "Act")

"OPTIMIZATION"

optimize_models(trees, depths, dataset_name, "drug_opt_results.txt"; time_limit=7200)