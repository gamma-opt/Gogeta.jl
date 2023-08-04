loaded_model = EvoTrees.load(string(@__DIR__)*"/trained_models/Concrete_Data.csv_1000_trees_9_depth.bson");
universal_model = extract_evotrees_info(loaded_model; tree_limit=500+1);

x_new, m_new, cons, creation_time, opt_time = tree_model_to_MIP(universal_model; create_initial=false, objective=MAX_SENSE, gurobi_env=ENV, timelimit=7200);
x_old, m_old, cons, creation_time, opt_time = tree_model_to_MIP(universal_model; create_initial=true, objective=MAX_SENSE, gurobi_env=ENV, timelimit=7200);

objective_value(m_new)
EvoTrees.predict(loaded_model, reshape([mean(x_new[n]) for n in 1:8], 1, 8); ntree_limit=500+1)[1]

objective_value(m_old)
EvoTrees.predict(loaded_model, reshape([mean(x_old[n]) for n in 1:8], 1, 8); ntree_limit=500+1)[1]

for tree in 1:universal_model.n_trees
    old_leaves = findall(pred->pred!=0, vec(universal_model.predictions[tree]))
    new_leaves = universal_model.leaves[tree]
    if !isempty(setdiff(old_leaves, new_leaves))
        println("Tree: $tree, In old, not in new: ", setdiff(old_leaves, new_leaves))
    end
    if !isempty(setdiff(new_leaves, old_leaves))
        println("Tree: $tree, In new, not in old: ", setdiff(new_leaves, old_leaves))
    end
end