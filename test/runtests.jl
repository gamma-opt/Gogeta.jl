using Gogeta
using Test
    
@testset "Neural networks" begin
    include("neural_networks/NN_test.jl")
end

@testset "Parallel tests" begin
    include("neural_networks/NN_parallel_test.jl")
end

@testset "Sampling tests" begin
    include("neural_networks/NN_sampling_test.jl")
end

@testset "Partition tests" begin
   include("neural_networks/NN_partition_test.jl")
end

@testset "Larger formulation tests" begin
    include("neural_networks/NN_in_larger_problem.jl")
end

@testset "ICNN tests" begin
    include("icnns/ICNN_in_larger_problem.jl")
end

@testset "CNN tests" begin
    include("neural_networks/CNN_test.jl")
end

@testset "Tree ensemble tests" begin
    include("tree_ensembles/TE_test.jl")
end
