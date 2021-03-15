using Flux, PGFPlotsX, DelimitedFiles
using BSON: @load
using Statistics
using LaTeXStrings

include("data.jl")
include("eval.jl")

@load "/home/oppenheimer/Dev/calabiyau/trained/euler/1/models/model-2021-03-14T15:48:37.637-CLASS_NUM-961-acc-29.332.bson" model
NUMBERS_ONLY = false
CLASS_NUM = 961
SAMPLE_SIZE = 1_000_000
DIM = 4
ontology_index = 4
MAX_SIZE = 961
train_set, test_set = import_split("/home/oppenheimer/Dev/calabiyau/trained/euler/1/data")
augment!(test_set)
test_x, test_y = unpackage(test_set)
test_y = euler_to_index(test_y)
pred = map(argmax, eachcol(model(test_x)))
# gr()

struct PlotParams
    title
    x_axis
    y_axis
end 

defaultParams = PlotParams("title", "x-axis", "y-axis")

## Confusion matrix    
function confusion_plot(k::Integer, 
                       ground_truth, 
                       prediction; 
                       bound = CLASS_NUM - 1, 
                       range = nothing,
                       params = defaultParams)

    cm = confusion_matrix(k, ground_truth, prediction)
    if ontology_index == 4
        a = euler_to_index(-bound)
        b = euler_to_index(bound)
    elseif ontology_index == 2 || ont == 3
        a = hodge_to_index(-bound)
        b = hodge_to_index(bound)
    else
        a = size(cm,1)
        b = size(cm,2)
    end
    cm = cm[a:b, a:b]
    x = [x for x in a:b]
    x = map(x->2(x-1)-960, x)
    y = x;
    if !(range ≡ nothing)
        return heatmap(x, y, cm, clim=range)
    else    
        return heatmap(x, y, cm)
    end
end

function confusion_waveplot(gts, pred; x_min=-960, x_max=960,y_max=961,z_max=1000, azimuth=15, elevation=35)
    predict_bit = [[x==ontology for x ∈ pred] for ontology ∈ 1:CLASS_NUM]
    gts = [test_y[predict_bit[ont]] for ont ∈ 1:CLASS_NUM]
    valid_euler = [!isempty(x) for x in gts] 
    gts = gts[valid_euler]
    pred_freq = [[count(x->x==class, gts[i]) for class ∈ 1:CLASS_NUM] for i ∈ 1:size(gts,1)]
    pred_freq = [pred_freq[i][clamp(euler_to_index(2*x_min),1,961):clamp(euler_to_index(2*x_max),1,961)] for i = 1:size(pred_freq,1)]
    # define the Axis to which we will push! the contents of the plot
    axis = @pgf Axis(
        {
            width = raw"1\textwidth",
            height = raw"0.6\textwidth",
            grid = "both",
            xmax = x_max,
            xmin = x_min,
            zmin = 0,
            # zmax = 50,
            "axis background/.style" = { fill = "gray!10" }, # add some beauty
            "every tick label/.style" = {font = raw"\tiny" },
            # this is needed to make the scatter points appear behind the graphs:
            set_layers,
            view = "{$azimuth}{$elevation}",   # viewpoint
            ytick = index_to_euler.(collect(1:50:961)),
            ztick = collect(0:500:4000)
        },
    )
    
    @pgf for i in eachindex(pred_freq)
        # if i%2 == 0
        curve = Plot3(
            {
                no_marks,
                style = {thin},
                color = "black"
            },
            Table(x = collect(LinRange(x_min,x_max,length(pred_freq[i]))),
                  y = index_to_euler(i) .* ones(length(pred_freq[i])),
                  z = smooth(pred_freq[i],3))
        )

        # The fill is drawn seperately to handle the the end of the curves nicely.
        # This is an alternative to "\fillbetween"
        fill = Plot3(
            {
                draw = "none",
                fill = "white",
                fill_opacity = 1.0
            },
            Table(x = collect(LinRange(x_min,x_max,length(pred_freq[i]))),
                  y = index_to_euler(i) .* ones(length(pred_freq[i])),
                  z = smooth(pred_freq[i],7))
        )
        push!(axis, curve)#, fill)
        # end
    end
    pgfsave("confusion_waveplot.pdf", axis)
    return true
end

## Error 
function error_plot(ground_truth, 
                   prediction, 
                   err; 
                   style = :scatter,
                   size = 100, 
                   x_bounds = (1, CLASS_NUM), 
                   y_bounds = (0,1), 
                   total_error = false, 
                   x_transform = identity,
                   label_x = 0.95,
                   label_y = 0.95,
                   params = defaultParams)

    error, μₑ, σₑ = error_stats(ground_truth, prediction, err)
    if total_error == true
        p_bool = [[x==ontology for x ∈ prediction] for ontology ∈ 1:CLASS_NUM]
        error = [sum(x) for x ∈ error[p_bool]]
    end
    label_x = x_bounds[2] * label_x
    label_y = y_bounds[2] * label_y
    label = latexstring("\\mu = $μₑ 
                         \\sigma = $σₑ")
    plot(x_transform.(prediction), 
         error, 
         st=style, 
         dpi=size, 
         xlim = x_bounds, 
         ylim = y_bounds, 
         markersize = 1,
         markercolor=:black, 
         annotations=(label_x, label_y, label),
         title = params.title,
         xaxis = params.x_axis,
         yaxis = params.y_axis,
         legend = false)
end

## Distributions
function data_plot(data; 
                          x_bound= CLASS_NUM - 1, 
                          params = defaultParams, 
                          save_figure = false)
    x = -960:2:960
    classes = [x[2] for x ∈ data]
    number_density = [count(x->x==i, classes) for i ∈ -960:2:960]

    p = plot(x,
             number_density, 
             seriestype = :bar, 
             dpi = 200, 
             markercolor= :red, 
             markersize = 0.25, 
             legend=false, 
             title=params.title, 
             xaxis=params.x_axis, 
             yaxis=params.y_axis)

    if save_figure == true
        savefig("../figs/test_dist.png")
    end
    return p
end

using Random
using Distributions


function test()
    Random.seed!(42)
    #Generate Data
    x_min = -10 # xrange to plot
    x_max = 10
    μ_min = -5
    μ_max = 5
    
    dist = (μ, σ) -> Normal(μ, σ)
    # make the set of distributions we're going to plot:
    dists = [dist(-6+sqrt(i), sqrt(1+0.3*i)) for i in 1:20]
    # creates random scatter points:
    rnd = rand.(Truncated.(dists, x_min, x_max), 20)
    # get the pdf of the dists:
    dat_pdf = [(x) -> pdf.(d, x) for d in dists]
    
    # point density for pdfs
    x_pnts = collect(x_min:2:x_max)
    
    # add redundant points at the ends, for nicer fill:
    x_pnts_ext = [[x_pnts[1]]; x_pnts; [x_pnts[end]]]
    
    # define the Axis to which we will push! the contents of the plot
    axis = @pgf Axis(
        {
            width = raw"1\textwidth",
            height = raw"0.6\textwidth",
            grid = "both",
            xmax = x_max,
            xmin = x_min,
            zmin = 0,
            "axis background/.style" = { fill = "gray!10" }, # add some beauty
            # this is needed to make the scatter points appear behind the graphs:
            set_layers,
            view = "{15}{35}",   # viewpoint
            ytick = collect(0:19),
            ztick = collect(0:0.1:1)
        },
    )
    
    # draw a yellow area at the bottom of the plot, centered at μ and 2σ wide.
    @pgf area = Plot3(
        {
            no_marks,
            style ="{dashed}",
            color = "black",
            fill = "yellow!60",
            fill_opacity = 0.65,
            # so we can see the grid lines through the colored area:
            on_layer = "axis background"
        },
        Table(x = [dists[1  ].μ - dists[1  ].σ, dists[end].μ - dists[end].σ,
                   dists[end].μ + dists[end].σ, dists[1  ].μ + dists[1  ].σ],
              y = [length(rnd) - 1, 0, 0, length(rnd) - 1],
              z = [0, 0, 0, 0]
             ),
        raw"\closedcycle"
    )
    push!(axis, area)
    
    # add the slices as individual plots to the common axis
    @pgf for i in eachindex(dists)
        scatter = Plot3(
            {
                only_marks,
                color = "red!80",
                mark_options = {scale=0.4},
                # set the markers on the same layer as the plot:
                mark_layer = "like plot",
                on_layer = "axis background"
            },
            Table(x = rnd[i],
                  y = (length(dists) - i) * ones(length(rnd[i])),
                  z = zeros(length(rnd[i])))
        )
        push!(axis, scatter)
    
        # add a pdf-curve on top of each second data set
        # if i%2 == 1
            curve = Plot3(
                {
                    no_marks,
                    style = {thin},
                    color = "black"
                },
                Table(x = x_pnts,
                      y = (length(dists) - i) * ones(length(x_pnts)),
                      z = dat_pdf[i](x_pnts))
            )
    
            # The fill is drawn seperately to handle the the end of the curves nicely.
            # This is an alternative to "\fillbetween"
            fill = Plot3(
                {
                    draw = "none",
                    fill = "white",
                    fill_opacity = 1.0
                },
                Table(x = x_pnts_ext,
                      y = (length(dists) - i) * ones(length(x_pnts_ext)),
                      z = [[0]; dat_pdf[i](x_pnts); [0]])
            )
            push!(axis, curve, fill)
        # end
    end
    pgfsave("test.pdf", axis)
end