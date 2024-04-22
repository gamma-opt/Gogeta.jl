# Tree ensembles

## Formulation

First, one must create and train an `EvoTrees` tree ensemble model.

```julia
using EvoTrees

config = EvoTreeRegressor(nrounds=500, max_depth=5)
evo_model = fit_evotree(config; x_train, y_train)
```

Then the parameters can be extracted from the trained tree ensemble and used to create a `JuMP` model containing the tree ensemble MIP formulation.

```julia
using Gurobi
using Gogeta

# Extract data from EvoTrees model

universal_tree_model = extract_evotrees_info(evo_model)

# Create jump model and formulate
jump = Model(() -> Gurobi.Optimizer())
set_attribute(jump, "OutputFlag", 0) # JuMP or solver-specific attributes can be changed

TE_formulate!(jump, universal_tree_model, MIN_SENSE)
```

## Optimization

There are two ways of optimizing the JuMP model: either by 1) creating the full set of split constraints before optimizing, or 2) using lazy constraints to generate only the necessary ones during the solution process.

1\) Full set of constraints

```julia
add_split_constraints!(jump, universal_tree_model)
optimize!(jump)
```

2\) Lazy constraints

```julia
# Define a callback function. For each solver this might be slightly different.
# See JuMP documentation or your solver's Julia interface documentation.
# Inside the callback 'tree_callback_algorithm' must be called.

function split_constraint_callback_gurobi(cb_data, cb_where::Cint)

    # Only run at integer solutions
    if cb_where != GRB_CB_MIPSOL
        return
    end

    Gurobi.load_callback_variable_primal(cb_data, cb_where)
    tree_callback_algorithm(cb_data, universal_tree_model, jump)

end

jump = direct_model(Gurobi.Optimizer())
TE_formulate!(jump, universal_tree_model, MIN_SENSE)

set_attribute(jump, "LazyConstraints", 1)
set_attribute(jump, Gurobi.CallbackFunction(), split_constraint_callback_gurobi)

optimize!(jump)
```

The optimal solution (minimum and maximum values for each of the input variables) can be queried after the optimization.

```julia
get_solution(opt_model, universal_tree_model)
objective_value(opt_model)
```

## Recommendations

Using the tree ensemble optimization from this package is quite straightforward. The only parameter the user can change is the solution method: with initial constraints or with lazy constraints.
In our computational tests, we have seen that the lazy constraint generation almost invariably produces models that are computationally easier to solve. 
Therefore we recommend primarily using it as the solution method, but depending on your use case, trying the initial constraints might also be worthwhile.