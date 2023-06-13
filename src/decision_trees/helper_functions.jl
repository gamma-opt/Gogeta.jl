function concrete_data()

    xf = XLSX.readxlsx("decision_trees/data/Concrete_Data.xlsx")
    data = xf["Sheet1"]

    x_train = Float64.([data["A2:A800"] data["B2:B800"] data["C2:C800"] data["D2:D800"] data["E2:E800"] data["F2:F800"] data["G2:G800"] data["H2:H800"]])
    y_train = Float64.(vec(data["I2:I800"]))

    x_test = Float64.([data["A801:A1031"] data["B801:B1031"] data["C801:C1031"] data["D801:D1031"] data["E801:E1031"] data["F801:F1031"] data["G801:G1031"] data["H801:H1031"]])
    y_test = Float64.(vec(data["I801:I1031"]))

    return x_train, y_train, x_test, y_test

end

function random_data()
    
    Random.seed!(3)

    nobs, nfeats = 1_000, 5
    x_train = randn(nobs, nfeats)
    y_train = Array{Float64}(undef, nobs)
    [y_train[i] = sum(x_train[i,:].^2) for i = 1:nobs]

    x_test = randn(nobs, nfeats)
    y_test = Array{Float64}(undef, nobs)
    [y_test[i] = sum(x_test[i,:].^2) for i = 1:nobs]

    return x_train, y_train, x_test, y_test

end

function build_forest(depth, n_trees, data_func)
    x_train, y_train, x_test, y_test = data_func()

    config = EvoTreeRegressor(max_depth=depth, nbins=32, nrounds=n_trees, loss=:linear, T=Float64)
    evo_model = fit_evotree(config; x_train, y_train)
    preds = EvoTrees.predict(evo_model, x_test)
    avg_error = rms(preds, y_test) / mean(y_test)

    return evo_model, preds, avg_error
end

function plot_model_quality(x, y, plot_title, label_x, label_y, data_func)

    z = Matrix{Float64}(undef, length(x), length(y))
    for xi in eachindex(x), yi in eachindex(y)
        evo_model, preds, avg_error = build_forest(y[yi], x[xi], data_func)
        z[xi, yi] = avg_error
    end
    
    # Interpolatant object
    itp = LinearInterpolation((x, y), z)
    # Fine grid
    x2 = range(extrema(x)..., length=300)
    y2 = range(extrema(y)..., length=200)
    # Interpolate
    z2 = [itp(x,y) for y in y2, x in x2]
    # Plot
    p = heatmap(x2, y2, z2, xlabel=label_x, ylabel=label_y, clim=(minimum(z), maximum(z)), title=plot_title)
    #scatter!(p, [x for _ in y for x in x], [y for y in y for _ in x], zcolor=z[:]; lab="original data") 
end