using Gogeta
using Test

@testset "Gogeta.jl" begin

    println("\n\n####################")
    println("Neural network tests")
    println("####################\n\n")
    
    # tests for neural networks
    include("neural_networks/NN_test.jl")
    
    println("\n\n####################")
    println("Neural network parallel tests")
    println("####################\n\n")
    
    # tests for neural networks
    include("neural_networks/NN_parallel_test.jl")
    
    println("\n\n###################")
    println("Tree ensemble tests")
    println("###################\n\n")

    # tests for tree ensembles
    include("tree_ensembles/TE_test.jl")
end
