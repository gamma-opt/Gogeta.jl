#=  exract_tree_nodes_info(evo_tree) is a function that extracts the information about all the nodes in the tree 
    Parameters:
        evo_tree: the trained tree from the set of tree forests (Support only for EvoTrees)
    Output: 
        tree: a collection of nodes from evo_tree (with detailed information)
=#
function exract_tree_nodes_info(evo_tree)

    split_nodes = findall(x->x!=0, evo_tree.split) # extract all the split nodes
    leaf_nodes = getindex.(findall(x->x!=0, evo_tree.pred), 2) # extract all the leaf nodes
    
    
    num_all_nodes = length(evo_tree.split) # extract the number of nodes in the complete tree*
    tree = Array{tree_node}(undef, num_all_nodes) # build an empty strucutree to keep info for each node
    
    for node_id = 1:num_all_nodes
        pruned = false # assume the node is not pruned
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

# complete tree* reffers to the tree of the maximum size (maximum number of nodes)