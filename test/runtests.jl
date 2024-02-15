using Gogeta
using Test

@testset "Gogeta.jl" begin
    # tests for neural networks
    #include("nn/DNN_test.jl")
    #nclude("nn/DNN_bound_tightening_test.jl")
    #include("nn/CNN_test.jl")
    
    # tests for neural networks
    include("neural_networks/NN_test.jl")
    
    # tests for tree ensembles
    include("tree_ensembles/TE_test.jl")
end
