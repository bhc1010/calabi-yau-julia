using Flux, PGFPlotsX, DelimitedFiles
using BSON: @load
using Statistics
using LaTeXStrings
cd("src")
include("data.jl")
include("eval.jl")

@load "/home/oppenheimer/Dev/calabiyau/trained/euler/1/models/model-2021-03-14T15:48:37.637-CLASS_NUM-961-acc-29.332.bson" model
NUMBERS_ONLY = false
CLASS_NUM = 961
SAMPLE_SIZE = 1_000_000
DIM = 4
ontology_index = 4
MAX_SIZE = 34
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
                       params = defaultParams,
                       save_figure = false)

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

    axis = @pgf Axis(
        {
            view = (0, 90),
            colorbar,
            "colormap/jet",
        },
        Plot3(
            {
                surf,
                shader = "flat",
            },
            Coordinates(x, y, cm)
        )
    )
    if save_figure == true
        pgfsave("../figs/confusion_matrix.pdf", axis)
    end
    return nothing
end

function confusion_waveplot(gts, pred;min=-960, max=960, azimuth=15, elevation=35, save_figure=false)
    predict_bit = [[x==ontology for x ∈ pred] for ontology ∈ 1:CLASS_NUM]
    gts = [test_y[predict_bit[ont]] for ont ∈ 1:CLASS_NUM]
    valid_euler = [!isempty(x) for x in gts]
    valid_euler = collect(-960:2:960)[valid_euler]
    # gts = gts[valid_euler]
    pred_freq = [[count(x->x==class, gts[i]) for class ∈ 1:CLASS_NUM] for i ∈ 1:size(gts,1)]
    # pred_freq = [pred_freq[i][clamp(euler_to_index(2*x_min),1,961):clamp(euler_to_index(2*x_max),1,961)] for i = 1:size(pred_freq,1)]
    # define the Axis to which we will push! the contents of the plot
    axis = @pgf Axis(
        {
            width = raw"1\textwidth",
            height = raw"0.6\textwidth",
            grid = "both",
            xmax = min,
            xmin = max,
            ymin = min,
            ymax = max,
            zmin = 0,
            # zmax = 5000,
            "axis background/.style" = { fill = "gray!10" }, # add some beauty
            "every tick label/.style" = {font = raw"\tiny" },
            # this is needed to make the scatter points appear behind the graphs:
            set_layers,
            view = "{$azimuth}{$elevation}",   # viewpoint
            ytick = index_to_euler.(collect(1:50:961)),
            ztick = collect(0:500:4000)
        },
    )
    
    @pgf for i in 961:-1:1
        if index_to_euler(i) ∈ valid_euler
            curve = Plot3(
                {
                    no_marks,
                    # mesh,
                    style = {thin},
                    color = "black"
                    # "colormap/jet"
                },
                Table(x = collect(LinRange(x_min,x_max,length(pred_freq[i]))),
                    y = index_to_euler(i) * ones(length(pred_freq[i])),
                    z = pred_freq[i])
            )

            # The fill is drawn seperately to handle the the end of the curves nicely.
            # This is an alternative to "\fillbetween"
            fill = Plot3(
                {
                    draw = "none",
                    fill = "white",
                    fill_opacity = 1.0
                },
                Table(x = [[x_min];collect(LinRange(x_min,x_max,length(pred_freq[i]))); [x_max]],
                    y = [[index_to_euler(i)];index_to_euler(i) * ones(length(pred_freq[i]));[index_to_euler(i)]],
                    z = [[pred_freq[i][1]];pred_freq[i];[pred_freq[i][end]]])
            )
            push!(axis, curve, fill)
        end
    end
    if save_figure == true
        pgfsave("../figs/confusion_waveplot.pdf", axis)
    end
    return nothing
end

## Error 
function error_plot(ground_truth, 
                   prediction, 
                   err; 
                   params = defaultParams,
                   save_figure = false)

    error, μₑ, σₑ = error_stats(ground_truth, prediction, err)
    axis = @pgf Axis(
        {
            title=params.title,
            xlabel=params.x_axis,
            ylabel=params.y_axis,
            # width="5in",
            xmin=-960,
            xmax=960
        },
        Plot(
            {
                "only marks",
                mark_size = "0.6pt",
                solid,
                color => "black"
            },
            Table(;
                x=index_to_euler(ground_truth),
                y=error
            )
        )
    )       
    if save_figure == true
        pgfsave("../figs/error_plot.pdf", axis)
    end
end

## Distributions
function data_plot(data; 
                   x_bound= CLASS_NUM - 1, 
                   params = defaultParams, 
                   save_figure = false)
    x = -960:2:960
    classes = [index_to_euler(x) for x ∈ data]
    number_density = [count(x->x==i, classes) for i ∈ -960:2:960]
    bins = [(x,y) for (x,y) ∈ zip(collect(x), number_density)]
    axis = @pgf Axis(
        {
            # width = "6in",
            title=params.title,
            xlabel=params.x_axis,
            ylabel=params.y_axis,
            xmin=-300,
            xmax = 300,
            ymin = 0,
            xtick = collect(-900:150:900),
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
    if save_figure == true
        pgfsave("../figs/test_dist.pdf", axis)
    end
end 