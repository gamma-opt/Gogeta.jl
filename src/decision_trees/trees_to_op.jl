#= GBtree_MIP(evo model) is a function that converts given gradient boosted tree ensemble to a MIP problem according to Music: Optimization of tree ensembles
    Parameters:
        evo_model: EvoTree{} - coontains the trained trees in field `trees` (incl. 1 bias tree) and other info accessible in field `info`(Support only for EvoTrees)
    Output: 
        model: a correspodning MIP 
=#
function GBtrees_MIP(evo_model)
    
    number_of_trees = length(evo_model.trees)-1 # number of tree in the model (the first element is not a tree)
    nfeats = length(evo_model.info[:fnames]) # number of features used

    # create an array that contains the number of leaves (not pruned) for each tree
    number_of_leaves = Array{Int64}(undef, number_of_trees)
    # the array that contains the ids of the nodes in the complete tree* that happpen to be leaves (for each tree)
    leaves = Array{Array}(undef, number_of_trees) 

    for t = 1:number_of_trees
        leaves[t] = findall(x->x!=0, vec(evo_model.trees[t+1].pred))
        @show leaves[t]
        number_of_leaves[t] = length(leaves[t])
    end

    # create an array with the list of splits for each feature (with the correspodning tree, node and split value identificator)
    splits = Array{Any}(undef, nfeats)  
    n_splits = Vector{Int64}(undef, nfeats) # total number of splits for each feature
    for f = 1:nfeats
        f_split_node = Vector{Int64}() # intialise empty vector for feature f to keep the split nodes value
        f_split_value = Vector{Float64}() # intialise empty vector for feature f to keep the split value
        f_split_tree = Vector{Int64}()  # intialise empty vector for feature f to keep the split trees value
        for t = 1:number_of_trees
            indices = findall(x->x==f, evo_model.trees[t+1].feat) # find the splits in tree t done on the feature f
            append!(f_split_node, indices)
            append!(f_split_value, evo_model.trees[t+1].cond_float[indices])
            append!(f_split_tree, t .* ones(length(indices)))
        end
        splits[f] = [f_split_tree'; f_split_node'; f_split_value'] # create a matrix for feature f [tree; node; split value]
        splits[f] = splits[f][:,sortperm(splits[f][3,:])] # sort the matix columns based on the 3rd row -> splits values 
        n_splits[f] = size(splits[f], 2) # store the number of splits in the whole forest associated with each feature 
    end

 
    model = Model(Gurobi.Optimizer)  # create an empty model
    @variable(model, x[f = 1:nfeats, 1:n_splits[f]], Bin ) 
    @variable(model, y[t = 1:number_of_trees, 1:number_of_leaves[t]]>=0) 
    @constraint(model, [i = 1:nfeats, j = 1:n_splits[i]-1], x[i,j] <= x[i, j+1])
    @constraint(model, [t = 1:number_of_trees], sum(y[t,l] for l = 1:number_of_leaves[t]) == 1) 

    for t = 1:number_of_trees
        splits_t_node_id = Vector{Int64}() # an array to store the split id value in the complete tree* 
        splits_t_node_feat = Vector{Int64}() # an array to store the split featre 
        splits_t_node_num = Vector{Int64}() # an array to store the split index in the array of all the splits on some feature
        for f = 1:nfeats
            # gather the inidices of all splits on feature f that are in the tree t (from the array of all the splits on feature f)
            split_nodes_t_f_ids = findall(x->x==t, splits[f][1, :]) 
            append!(splits_t_node_id, splits[f][2, split_nodes_t_f_ids])
            append!(splits_t_node_feat, f .* ones(length(split_nodes_t_f_ids )))
            append!(splits_t_node_num, split_nodes_t_f_ids)
        end
        for s in 1:length(splits_t_node_id)
            # generate the left/right branch leaves from the split s in the complete tree*
            left_children, right_children = find_left_right_leaves(splits_t_node_id[s], evo_model.trees[t+1].pred) 
            # an array to store indices of leaves in y[t,:] that correspond to left branch children nodes ids in complete tree*
            y_indices_left = Vector{Int64}() 
            for c = 1:length(left_children)
                index = findall( x -> x == left_children[c], leaves[t])
                append!(y_indices_left, index)
            end
            @constraint(model, sum(y[t,i] for i in y_indices_left) <= x[splits_t_node_feat[s], splits_t_node_num[s]])
            # an array to store indices of leaves in y[t,:] that correspond to right branch children nodes ids in complete tree*
            y_indices_right = Vector{Int64}()
            for c = 1:length(right_children)
                index = findall( x -> x == right_children[c], leaves[t])
                append!(y_indices_right, index)
            end
            @constraint(model, sum(y[t,i] for i in y_indices_right) <= 1 - x[splits_t_node_feat[s], splits_t_node_num[s]])
        end
    end

    @objective(model, Min, sum(0.1 * evo_model.trees[t+1].pred[leaves[t][l]] * y[t,l] for t = 1:number_of_trees, l = 1:number_of_leaves[t]))

    optimize!(model)

    print_solution(nfeats, model, n_splits, splits)

    return model

end

# complete tree* reffers to the tree of the maximum size (maximum number of nodes)