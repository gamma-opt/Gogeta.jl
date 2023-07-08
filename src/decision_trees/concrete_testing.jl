"DATA LOADING"

dataset_name = "Concrete_Data.csv";
train_data, x_train, y_train, x_test, y_test, feat_names = load_concrete_data(dataset_name);

"TREE MODEL GENERATION"

trees = [10, 50, 100, 200, 350, 500, 750, 1000];
depths = [3, 5, 7, 9, 12];

train_evo_models(depths, trees, train_data, feat_names, x_train, y_train, x_test, y_test, "concrete_test_results.txt", dataset_name, "Concrete compressive strength")

"OPTIMIZATION"

optimize_models(trees, depths, dataset_name, "concrete_opt_results.txt"; time_limit=1000)