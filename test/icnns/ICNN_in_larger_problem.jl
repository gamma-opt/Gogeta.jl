using Random
using Gogeta
using GLPK
using JuMP

@info "creating an optimization problem"
jump_model = Model(GLPK.Optimizer)

@variable(jump_model, -1 <= x <= 1)
@variable(jump_model, -1 <= y <= 1)
@variable(jump_model, output)

@constraint(jump_model, y >= 1-x)

@objective(jump_model, Min, x+y)

@info "include input convex neural network as a part of the larger optimization problem"
ICNN_incorporate!(jump_model, "icnns/model_weights.json", output, x, y)

optimize!(jump_model)
solution_summary(jump_model)

@info "testing for correct values"
@test value(x) ≈ 0.43243156544161554
@test value(y) ≈ 0.5675684345583831
@test value(output) ≈ 0.5993355385172512