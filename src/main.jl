include("initialisation.jl")


config = EvoTreeRegressor(max_depth=5, nbins=32, nrounds=10)
nobs, nfeats = 1_000, 5
x_train, y_train = randn(nobs, nfeats), rand(nobs)
model = fit_evotree(config; x_train, y_train)
preds = EvoTrees.predict(model, x_train)
plot(model, 3)


evo_tree = model.trees[3]

mutable struct tree_node
    id:: Int64 # node id
    pruned:: Bool # the information about whether the node is pruned 
    type:: Int64 # the type of the node 1 - split, 2 - leaf
    left_node:: Int64 # id of the node of the left branch if applicable (otherwise 0)
    right_node:: Int64 # id of the node of the right branch if applicable (otherwise 0)
    s_feat:: Int64 # feature id on which the split is made if applicable (otherwise 0)
    s_value:: Float64 # the value of a for the split condition x<=a if applicable (otherwise 0)
    f_value:: Float64 # the value of the predicition of a leaf node if applicable (otherwise 0)
    leaves_left:: Array{Int64} # list of leaves ids accessable from the left branch if applicable (otherwise [0])
    leaves_right:: Array{Int64} # list of leaves ids accessable from the right branch if applicable (otherwise [0])
end 

#=  find_left_right_leaves(node_id, evo_tree_pred) is a function that find the leaves that are accesable 
                                                from the left and from the right branches of the node 
    Parameters:
        node_id: id of the node under investigation 
        evo_tree_pred: the list of the predicitions for all the nodes of the tree 
        *the predicitons are zero for split or pruned nodes    
    Output: 
        left, rigt : two arrays that containt the ids of leaves accesable from the left and right split branches repectively
        * if the node with node_id is a leaf or pruned node then left and right will be empty arrays 
=#
function find_left_right_leaves(node_id, evo_tree_pred)
    left = [] # leaves that are accesable from the left branch 
    right = [] # leaves that are accesable from the left branch 
    current_id = node_id
    if 2*node_id <= length(evo_tree_pred) # if the node has children
        current_left_child = current_id*2
        current_right_child = current_id*2+1
        if evo_tree_pred[current_left_child]!=0 # if the child is already leaf
            push!(left, current_left_child) # stop further search
        else 
            all_children_left = find_all_children(current_left_child, length(evo_tree_pred), []) # find all children
            for i = 1:length(all_children_left)
                if evo_tree_pred[all_children_left[i]]!=0 # if the child is leaf
                    push!(left, all_children_left[i]) # add the child
                end
            end
        end

        if evo_tree_pred[current_right_child]!=0 # if the child is already leaf
            push!(right, current_right_child) # stop further search
        else 
            all_children_right = find_all_children(current_right_child, length(evo_tree_pred), []) # find all children
            for i = 1:length(all_children_right)
                if evo_tree_pred[all_children_right[i]]!=0 # if the child is leaf
                    push!(right, all_children_right[i]) # add the child
                end
            end
        end        
    end
    return left, right
end 

#=  find_all_children(node_id, nodes_max, all_children) is a function that finds all the children nodes
    Parameters:
        node_id: id of the node under investigation 
        nodes_max: maximum number of nodes in the tree
        all_children: an array that should be initialized empty and will be recursively filled with children nodes
    Output: 
        all_children: an array that contains all children nodes
        * if the node_id node is leaf in the maxumum tree structure the all_children array will be empty
=#
function find_all_children(node_id, nodes_max, all_children)
    if 2*node_id <= nodes_max # if node has children
        push!(all_children, 2*node_id, 2*node_id+1)
        find_all_children(2*node_id, nodes_max, all_children)
        find_all_children(2*node_id+1, nodes_max, all_children)
    end
    return all_children
end



#=  exract_tree_nodes_info(evo_tree) is a function that extracts the information about all the nodes in the tree 
    Parameters:
        evo_tree: the trained tree from the set of tree forests (Support only for EvoTrees)
    Output: 
        tree: a collection of nodes from evo_tree (with detailed information)
=#
function exract_tree_nodes_info(evo_tree)

    split_nodes = findall(x->x!=0, evo_tree.split) #extract all the split nodes
    leaf_nodes = getindex.(findall(x->x!=0, evo_tree.pred), 2) #extract all the leaf nodes
    
    
    num_all_nodes = length(evo_tree.split) #extract the number of nodes in the maximum tree
    tree = Array{tree_node}(undef, num_all_nodes) #build an empty strucutree to keep info for each node
    for node_id = 1:num_all_nodes
        pruned = false #assume the node is not pruned
        if !isempty(findall(x->x==node_id, split_nodes)) # check if the node is a split node
            type = 1 
        elseif  !isempty(findall(x->x==node_id, leaf_nodes)) # check if the node is a leaf 
            type = 2
        else 
            pruned = true # assume the node pruned
        end 


        if pruned 
            node = tree_node(       
                            node_id ,
                            true,
                            0, 
                            0, 
                            0,
                            evo_tree.feat[node_id ],
                            evo_tree.cond_float[node_id ],
                            evo_tree.pred[node_id ],
                            [],
                            []
            ) # build a pruned node
        else 
            left_split_leaves, right_split_leaves = find_left_right_leaves(node_id, evo_tree.pred) # gather the information about the leaves from the left/roght split
            node = tree_node(   
                            node_id ,
                            false, 
                            type, 
                            type == 1 ? 2*node_id : 0,  # if the node is not a leaf 
                            type == 1 ? 2*node_id+1 : 0, # if the node is not a leaf 
                            evo_tree.feat[node_id ],
                            evo_tree.cond_float[node_id ],
                            evo_tree.pred[node_id ],
                            left_split_leaves,
                            right_split_leaves
            ) # build up a correspding node
        end
        tree[node_id] = node  
    end
    return tree
end
