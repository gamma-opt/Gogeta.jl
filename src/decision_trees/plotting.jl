function plot_model_quality(x, y, z, plot_title, label_x, label_y)

    # Interpolatant object
    itp = LinearInterpolation((x, y), z)
    # Fine grid
    x2 = range(extrema(x)..., length=3000)
    y2 = range(extrema(y)..., length=2000)
    # Interpolate
    z2 = [itp(x,y) for y in y2, x in x2]
    # Plot
    p = heatmap(x2, y2, z2, 
                xaxis=:log, 
                c=cgrad(:roma, scale=:log), 
                xlabel=label_x, 
                ylabel=label_y, 
                clim=(minimum(z), maximum(z)), 
                title=plot_title,
                xticks=(x, string.(x)),
                yticks=(y, string.(y)))
    #scatter!(p, [x for _ in y for x in x], [y for y in y for _ in x], zcolor=z[:]; lab="original data") 
end