# Bound tightening

To improve the computational feasibility of the mixed-integer formulation of a neural network, the big-M values associated with the some of the constraints can be made smaller by calculating the minimum and and maximum activations of the individual neurons. This can be done with a heuristic algorithm or by using mathematical optimization.

Our package includes three different modes of bound tightening: `fast` *(default)*, `standard` and `output`.
1. The `fast` mode uses a heuristic algorithm to determine the neuron activation bounds only based on the activation bounds of the previous layer. This algorithm practically doesn't increase the formulation time, so it is enabled by default. 
2. The `standard` mode considers the whole mixed-integer problem with variables and constraints defined up to the previous layer from the neuron under bound tightening. It uses optimization to find the neuron activation bounds and is therefore significantly slower than the `fast` mode but is able to produce tighter bounds (smaller big-M values).
3. In some situations, the user might know the bounds for the output layer neurons. The `output` mode takes into account these output bounds as well as the whole MIP. Therefore, it is able to produce the tightest bounds of all the methods listed, but it is also the slowest.

(`precomputed` is also one of the bound tightening options in the functions. It can be used by inputting bounds that have already been calculated.)

A detailed discussion on bound tightening techniques can be found in [Grimstad and Andresson (2019)](literature.md).