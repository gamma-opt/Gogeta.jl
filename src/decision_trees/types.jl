#the structure to keep the detailed informmation about the tree node
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