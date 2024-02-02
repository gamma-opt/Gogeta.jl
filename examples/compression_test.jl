data = rand(Float32, (2, 1000)) .- 0.5f0;
x_train = data[:, 1:750];

using Flux
using Random
using Revise
using Gogeta

begin
    Random.seed!(1234);

    model = Chain(
        Dense(2 => 10, relu),
        Dense(10 => 50, relu),
        Dense(50 => 20, relu),
        Dense(20 => 5, relu),
        Dense(5 => 1)
    )
end

begin
    Random.seed!(1234);

    model = Chain(
        Dense(2 => 10, relu),
        Dense(10 => 20, relu),
        Dense(20 => 5, relu),
        Dense(5 => 1)
    )
end

begin
    Random.seed!(1234);

    model = Chain(
        Dense(2 => 3, relu),
        Dense(3 => 1)
    )
end

solver_params = SolverParams(silent=true, threads=0, relax=false, time_limit=0)

@time jump, removed = compress(model, [1.0, 1.0], [-1.0, -1.0], solver_params);
vec(model(x_train)) ≈ [forward_pass!(jump, x_train[:, i])[1] for i in 1:750]

@time _, U_full, L_full = NN_to_MIP(model, [1.0, 1.0], [-1.0, -1.0], solver_params; tighten_bounds=true);

using Plots

plot(collect(Iterators.flatten(U[1:end-1])))
plot!(collect(Iterators.flatten(U_full[1:end-1])))
