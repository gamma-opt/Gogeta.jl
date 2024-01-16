using ML_as_MO
using Test

@testset "ML_as_MO.jl" begin
    include("CNN_test.jl")
    include("DNN_test.jl")
    include("DNN_bound_tightening_test.jl")
end


