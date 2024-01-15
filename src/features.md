# Package features

## Decision trees
* **tree ensemble optimization** - ability to optimize a trained decision tree model, i.e., to find an input that maximizes the forest output

### Interface
1. *extract_evotrees_info*
2. *tree_model_to_MIP*

## Neural networks
* **neural network to MILP conversion**
* **bound tightening** - single-threaded, multithreaded and distributed
* **neural network compression** - reduce network size by removing unnecessary nodes
* **generating adverserial examples** ???

### Interface
1. *create_JuMP_model*
2. *create_CNN_JuMP_model*
3. *bound_tightening*
4. *evaluate!*
5. *compress_network* ???