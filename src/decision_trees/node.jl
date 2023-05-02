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
