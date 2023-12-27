using Gogeta
using Test

@testset "Gogeta.jl" begin
    # tests for NN 
    include("nn/DNN_test.jl")
    include("nn/DNN_bound_tightening_test.jl")
    include("nn/CNN_test.jl")
    

    
    # Write your tests here.
end
