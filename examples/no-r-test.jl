using Revise
using Gogeta
using Flux
using Random
using Plots

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

init_U = [-0.5, 0.5];
init_L = [-1.5, -0.5];

x1 = (rand(100) * (init_U[1] - init_L[1])) .+ init_L[1];
x2 = (rand(100) * (init_U[2] - init_L[2])) .+ init_L[2];
x = transpose(hcat(x1, x2)) .|> Float32;

solver_params = SolverParams(solver="Gurobi", silent=true, threads=0, relax=false, time_limit=0);

@time jump_nor, U_nor, L_nor = NN_to_MIP(model, init_U, init_L, solver_params; tighten_bounds="output", out_ub=[-0.2], out_lb=[-0.4]);
@time jump_standard, U_standard, L_standard = NN_to_MIP(model, init_U, init_L, solver_params; tighten_bounds="standard");
@time jump_fast, U_fast, L_fast = NN_to_MIP(model, init_U, init_L, solver_params; tighten_bounds="fast");

@btime [forward_pass!(jump_nor, input)[] for input in eachcol(x)];

# Plot the differences
plot(collect(Iterators.flatten(U_standard)))
plot!(collect(Iterators.flatten(U_nor)))
plot!(collect(Iterators.flatten(U_fast)))

plot!(collect(Iterators.flatten(L_standard)))
plot!(collect(Iterators.flatten(L_nor)))
plot!(collect(Iterators.flatten(L_fast)))

x_range = LinRange{Float32}(init_L[1], init_U[1], 100);
y_range = LinRange{Float32}(init_L[2], init_U[2], 100);

contourf(x_range, y_range, (x, y) -> model(hcat(x, y)')[], c=cgrad(:viridis), lw=0)
contourf(x_range, y_range, (x, y) -> forward_pass!(jump_nor, [x, y])[] === nothing ? NaN : forward_pass!(jump_nor, [x, y])[], c=cgrad(:viridis), lw=0)