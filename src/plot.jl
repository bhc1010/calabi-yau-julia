using Flux, PGFPlotsX, DelimitedFiles
using BSON: @load
using Statistics

"""
PlotParams holds information needed for plotting, e.g. type of plot, title, x and y labels, directory to save figure, etc.
"""
mutable struct PlotParams
    plot::Function
    title
    x_label
    y_label
    model_label
    x_min
    x_max
    x_range
    num_xticks
    y_min
    y_max
    y_range
    num_yticks
    err
    fig_dir
    colorbar_label
end

"""
PlotParams custom constuctor
"""
function new_params(plot;
                     title=nothing,
                     x_label=nothing,
                     y_label=nothing,
                     model_label=nothing,
                     x_min=nothing,
                     x_max = nothing,
                     x_range = nothing,
                     num_xticks = nothing,
                     y_min = nothing,
                     y_max = nothing,
                     y_range = nothing,
                     num_yticks=nothing,
                     err=nothing,
                     fig_dir = nothing,
                     colorbar_label=nothing)
    return PlotParams(plot, title, x_label, y_label, model_label, x_min, x_max, x_range, num_xticks, y_min, y_max, y_range, num_yticks, err, fig_dir, colorbar_label)
end

"""
Generate plots for each set of PlotParams given in params argument.
"""
function make_plots(model, data, plots; load_data=false, load_model=false, η=identity)
    load_model ? (@load model m) : local m = model;

    if load_data
        train_set, test_set = import_split(data_dir)
        augment!(test_set)
        test_x, gt = unpackage(test_set)
        gt = euler_to_index(gt)
    else
        (test_x, gt) = data
    end
    pred = η(map(argmax, eachcol(m(test_x))))
    for p ∈ plots
        if isnothing(p.err)
            p.plot((gt, pred), params=p)
        else
            p.plot((gt, pred), err=p.err, params=p)
        end
    end
end

"""
Plot confusion matrix for data
"""
## Confusion matrix    
function plot_cmatrix( data;
                       bound = CLASS_NUM - 1,
                       k=nothing,
                       params = defaultParams,
                       fig_dir=nothing)
    
    !isnothing(params.x_range) && (bound = params.x_range)
    if isnothing(k)
        (ontology_index == 2) && (k = 480)
        (ontology_index == 3) && (k = 480)
        (ontology_index == 4) && (k = 961)
    end
    (gt, pred) = data
    if ontology_index == 4
        a = euler_to_index(-bound)
        b = euler_to_index(bound)
        gt = euler_to_index(gt)
        pred = euler_to_index(pred)
    else
        a = 1
        b = bound
    end
    cm = confusion_matrix(k, gt, pred)
    cm = cm[a:b, a:b]
    x = collect(a:b)
    y = x;

    plot_heatmap(x, y, cm; params=params)
end

"""
Plot a PGFPlots heatmap.
"""
function plot_heatmap(x, y, z; params=defaultParams)
    
    if !isnothing(params.fig_dir)
        fig_dir = params.fig_dir
    else
        @error("No figure directory given.")
        return nothing
    end

    axis = @pgf Axis(
        {
            title=params.title,
            xlabel=params.x_label,
            ylabel=params.y_label,
            view = (0, 90),
            colorbar,
            "colormap/jet",
            "point meta max = 1000",
            colorbar_style = {
                                ylabel=params.colorbar_label
                             },
        },
        Plot3(
            {
                surf,
                shader = "flat",
            },
            Coordinates(x, y, z)
        )
    )
    !isnothing(fig_dir) && pgfsave(fig_dir, axis)
end

"""
Plot error on data. Error calculated by error function argument (defaults to relative error).
"""
## Error 
function plot_error(data; 
                   err = rel_error,
                   params = defaultParams,
                   x_min = -960,
                   x_max = 960,
                   x_range = nothing,
                   num_xticks = 9,
                   y_range = nothing,
                   y_min = nothing,
                   y_max = nothing,
                   num_yticks= 9,
                   fig_dir=nothing)
    (gt, pred) = data
    error, μₑ, σₑ = error_stats(gt, pred, err)
    max_error = maximum(error)
    min_error = minimum(error)
    x_min, x_max, y_min, y_max, num_xticks, num_yticks = _maybe_useparams(params, x_min, x_max, y_min, y_max, num_xticks, num_yticks)
    (isnothing(fig_dir) && !isnothing(params.fig_dir)) && (fig_dir = params.fig_dir)
    isnothing(y_min) && (y_min = min_error)
    isnothing(y_max) && (y_max = max_error)
    !isnothing(x_range) && (x_min = -abs(x_range)) & (x_max = abs(x_range))
    !isnothing(y_range) && (y_min = -abs(y_range)) & (y_max = abs(y_range))
    axis = @pgf Axis(
        {
            title=params.title,
            xlabel=params.x_label,
            ylabel=params.y_label,
            xmin = x_min,
            xmax = x_max,
            xtick = collect(range(x_min, x_max; length=num_xticks)),
            ytick = collect(range(y_min, y_max; length=num_yticks)),
            ymin = y_min,
            ymax = y_max,
            "minor y tick num = 9",
            "every axis/.append style" = {font = raw"\footnotesize" },
            "every node" = {font = raw"2em"},
            no_marks,
            legend_style={nodes={"scale=0.5"}, draw="none"},
        },
        Plot(
            {
                "only marks",
                mark_size = "0.1pt",
                solid,
                color => "black"
            },
            Table(;
                x=gt,
                y=error
            )
        ),
        LegendEntry(params.model_label),
        Plot({no_marks, mark_size="0.1pt", color="white"},Table(;x=x_min,y=y_min)),LegendEntry("\\mu = $μₑ"),
        Plot({no_marks, mark_size="0.1pt", color="white"},Table(;x=x_min,y=y_min)),LegendEntry("\\sigma = $σₑ"),
    )       
    !isnothing(fig_dir) && pgfsave(fig_dir, axis)
end

"""
Plot error with colorbar. Color denotes data[2]
"""
## Error 
function plot3_error(data; 
                    err = rel_error,
                    params = defaultParams,
                    x_min = -960,
                    x_max = 960,
                    x_range = nothing,
                    num_xticks = 9,
                    y_max = nothing,
                    fig_name=nothing)
    colorbar_label = ""
    (gt,pred) = data
    if ontology_index == 4
        valid = index_to_euler(gt) .!= 0
        gt = index_to_euler(gt[valid])
        pred = index_to_euler(pred[valid])
        colorbar_label = "Predicted Euler"
    elseif ontology_index == 3 
        colorbar_label = "Predicted \$h^{2,1}\$"
    elseif ontology_index == 2
        colorbar_label = "Predicted \$h^{1,1}\$"
    end
    error, μₑ, σₑ = error_stats(gt, pred, err)
    !isnothing(x_range) && (x_min = -abs(x_range)) & (x_max = abs(x_range))
    axis = @pgf Axis(
        {
            title=params.title,
            xlabel=params.x_label,
            ylabel=params.y_label,
            xmin = x_min,
            xmax = x_max,
            xtick = collect(range(x_min, x_max; length=num_xticks+1)),
            ymin = 0,
            ymax = y_max,
            "minor y tick num = 3",
            raw"every axis/.append style" = {font = raw"\footnotesize" },
            view = "{0}{90}",
            colorbar,
            "colormap/jet",
            colorbar_style = {
                                ylabel=colorbar_label
                             },
            legend_style={nodes={"scale=0.5"}, draw="none"},
        },
        Plot3(
            {
                "only marks",
                scatter,
                mark_size = "0.1pt",
                solid,
                color => "black"
            },
            Table(;
                x=gt,
                y=error,
                z=pred
            )
        ),
        LegendEntry(params.model_label),
        Plot({only_marks, mark_size="0.1pt", color="white"},Table(;x=0,y=0)),LegendEntry("\\mu = $μₑ"),
        Plot({only_marks, mark_size="0.1pt", color="white"},Table(;x=0,y=0)),LegendEntry("\\sigma = $σₑ"),
    )       
    !isnothing(fig_name) && pgfsave("../../figs/$fig_name", axis)
end

function plot_error_vs_firstorder(data; 
                   err = rel_error,
                   params = nothing,
                   x_min = -960,
                   x_max = 960,
                   x_range = nothing,
                   num_xticks = 9,
                   y_range = nothing,
                   y_min = nothing,
                   y_max = nothing,
                   num_yticks= 9,
                   fig_dir=nothing)
    (gt, X, pred) = data
    fo = calculate_leadingpiece(X)
    error, μₑ, σₑ = error_stats(gt, pred, err)
    error_fo, μ_fo, σ_fo = error_stats(fo, pred, err)
    max_error = maximum(error)
    min_error = minimum(error)
    x_min, x_max, y_min, y_max, num_xticks, num_yticks = _maybe_useparams(params, x_min, x_max, y_min, y_max, num_xticks, num_yticks)
    (isnothing(fig_dir) && !isnothing(params.fig_dir)) && (fig_dir = params.fig_dir)
    isnothing(y_min) && (y_min = min_error)
    isnothing(y_max) && (y_max = max_error)
    !isnothing(x_range) && (x_min = -abs(x_range)) & (x_max = abs(x_range))
    !isnothing(y_range) && (y_min = -abs(y_range)) & (y_max = abs(y_range))
    axis = @pgf Axis(
        {
            title=params.title,
            xlabel=params.x_label,
            ylabel=params.y_label,
            xmin = x_min,
            xmax = x_max,
            xtick = collect(range(x_min, x_max; length=num_xticks)),
            ytick = collect(range(y_min, y_max; length=num_yticks)),
            ymin = y_min,
            ymax = y_max,
            "minor y tick num = 9",
            "every axis/.append style" = {font = raw"\footnotesize" },
            "every node" = {font = raw"2em"},
            no_marks,
            legend_style={nodes={"scale=0.5"}, draw="none"},
        },
        Plot(
            {
                "only marks",
                mark_size = "0.1pt",
                solid,
                color => "blue",
                opacity = 0.01,
            },
            Table(;
                x=gt,
                y=error
            )
        ),
        Plot(
            {
                "only marks",
                mark_size = "0.1pt",
                solid,
                color => "red",
                opacity = 0.01,
            },
            Table(;
                x=fo,
                y=error_fo
            )
        ),
    )       
    !isnothing(fig_dir) && pgfsave(fig_dir, axis, dpi=600)
end

"""
Plot data as a histogram 
"""
## Distributions
function plot_datahist(data, ontology;
                   x_bound= CLASS_NUM - 1, 
                   params = defaultParams,
                   fig_dir=nothing)
    (gt, p) = data
    if ontology == "euler"
        local o_range = -960:2:960
    elseif ontology == "h11" || ontology == "h21"
        local o_range = 1:480
        local x_ticks = [1,100,100,200,200,300,400,480]
    end
    number_density = [count(x->x==i, gt) for i ∈ o_range]
    bins = [(x,y) for (x,y) ∈ zip(collect(o_range), number_density)]
    (isnothing(fig_dir) && !isnothing(params.fig_dir)) && (fig_dir = params.fig_dir)
    axis = @pgf Axis(
        {
            title=params.title,
            xlabel=params.x_label,
            ylabel=params.y_label,
            xmin = params.x_min,
            xmax = params.x_max,
            ymin = params.y_min,
            xtick = x_ticks,
            grid = "none",
            "minor y tick num = 3",
        },
        Plot(
            {
                "mark=no",
                "ybar interval",
                fill = "black", 
                color = "black",
            }, 
            Coordinates(bins)
        )
    )  
    !isnothing(fig_dir) && pgfsave(fig_dir, axis)
end 

"""
Plot data as a histogram 
"""
## Distributions
function plot_datahist_with_prediction(data, ontology;
                   x_bound= CLASS_NUM - 1, 
                   params = defaultParams,
                   fig_dir=nothing)
    (gt, p) = data
    if ontology == "euler"
        local o_range = -960:2:960
    elseif ontology == "h11" || ontology == "h21"
        local o_range = 1:480
    end
    number_density_gt = [count(x->x==i, gt) for i ∈ o_range]
    bins_gt = [(x,y) for (x,y) ∈ zip(collect(o_range), number_density_gt)]
    number_density_p = [count(x->x==i, p) for i ∈ o_range]
    bins_p = [(x,y) for (x,y) ∈ zip(collect(o_range), number_density_p)]
    (isnothing(fig_dir) && !isnothing(params.fig_dir)) && (fig_dir = params.fig_dir)

    axis = @pgf Axis(
        {
            title=params.title,
            xlabel=params.x_label,
            ylabel=params.y_label,
            xmin = params.x_min,
            xmax = params.x_max,
            ymin = params.y_min,
            xtick = collect(-900:25:900),
            grid = "none",
            "minor y tick num = 3",
        },
        Plot(
            {
                "mark=no",
                "ybar interval",
                fill = "blue", 
                color = "blue",
                opacity = 0.5,
            }, 
            Coordinates(bins_gt)
        ),
        Plot(
            {
                "mark=no",
                "ybar interval",
                fill = "red", 
                color = "red",
                opacity = 0.5,
            }, 
            Coordinates(bins_p)
        )
    )  
    !isnothing(fig_dir) && pgfsave(fig_dir, axis)
end 

"""
Plot mirror symmetry of data: h₁,₁ + h₂,₁ vs h₁,₁ - h₂,₁
"""
function plot_mirror(data; 
                     params=defaultParams,
                     x_min=nothing,
                     x_max=nothing,
                     x_range = nothing,
                     num_xticks = 9,
                     y_range = nothing,
                     y_min = nothing,
                     y_max = nothing,
                     num_yticks= 9,
                     fig_dir=nothing)
    (ontology_index == 4) && (return @info("Ontology not set to hodge."))
    (h11,h21) = data
    xs = h11 .- h21 
    ys = h11 .+ h21
    (isnothing(fig_dir) && !isnothing(params.fig_dir)) && (fig_dir = params.fig_dir)

    # x_min, x_max, y_min, y_max, num_xticks, num_yticks = _maybe_useparams(params,x_min, x_max, y_min, y_max, num_xticks, num_yticks)
    axis = @pgf Axis(
        {
            title=params.title,
            xlabel=params.x_label,
            ylabel=params.y_label,
            # xmin = x_min,
            # xmax = x_max,
            # xtick = collect(range(x_min, x_max; length=num_xticks)),
            # ytick = collect(range(y_min, y_max; length=num_yticks)),
            # ymin = y_min,
            # ymax = y_max,
            "minor y tick num = 9",
            "every axis/.append style" = {font = raw"\footnotesize" },
            "every node" = {font = raw"2em"},
            no_marks,
            legend_style={nodes={"scale=0.5"}, draw="none"},
        },
        Plot(
            {
                "only marks",
                mark_size = "0.05pt",
                solid,
                fill => "black",
                color => "black"
            },
            Table(;
                x=xs,
                y=ys
            )
        ),
        LegendEntry(params.model_label),
    )
    !isnothing(fig_dir) && pgfsave(fig_dir, axis)
end

"""
_maybe_useparams decides which params to use
"""
function _maybe_useparams(params, xmin, xmax, ymin, ymax, numxticks, numyticks)
    x_min = xmin
    x_max = xmax 
    y_min = ymin 
    y_max = ymax 
    num_xticks = numxticks
    num_yticks = numyticks
    !isnothing(params.x_range) && (x_min = -abs(params.x_range)) & (x_max = abs(params.x_range))
    !isnothing(params.y_range) && (y_min = -abs(params.y_range)) & (y_max = abs(params.y_range))
    !isnothing(params.x_min) && (x_min = params.x_min)
    !isnothing(params.x_max) && (x_max = params.x_max)
    !isnothing(params.y_min) && (y_min = params.y_min)
    !isnothing(params.y_max) && (y_max = params.y_max)
    !isnothing(params.num_xticks) && (num_xticks = params.num_xticks)
    !isnothing(params.num_yticks) && (num_yticks = params.num_yticks)
    !isnothing(params.num_xticks) && (num_xticks = params.num_xticks)
    return x_min, x_max, y_min, y_max, num_xticks, num_yticks
end