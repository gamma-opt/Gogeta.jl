function plot_model_quality(x, y, z; plot_title="TITLE", label_x="X LABEL", label_y="Y LABEL", lim_l=minimum(z), lim_h=maximum(z))

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
                clim=(lim_l, lim_h), 
                title=plot_title,
                xticks=(x, string.(x)),
                yticks=(y, string.(y)))
    
    #scatter!(p, [x for _ in y for x in x], [y for y in y for _ in x], zcolor=z[:]; lab="original data")
end