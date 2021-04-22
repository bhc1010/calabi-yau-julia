@info("Precompiling headers...")
using Flux, Zygote, CUDA, Random
using Flux:@epochs
using Flux.Data:DataLoader
using Flux:onehotbatch
using Printf, BSON, Dates, DelimitedFiles

include("data.jl")
include("train.jl")
include("eval.jl")
include("plot.jl")

# Parameters for data I/O
TOTAL_DATA = 473_800_775
TOTAL_NUM_DB = 474
DIM = 4
SAMPLE_SIZE = 1_000_000
DB_PATH = "/media/share/Dev/CalabiYau/data/polytopes_db_4d"
PATH = "/home/oppenheimer/Dev/calabiyau/trained"
NUMBERS_ONLY = false

# Parameters for model building
MAX_SIZE = 34
ONTOLOGY = "euler"
if ONTOLOGY == "h11"
    ontology_index = 2
    CLASS_NUM = 480
elseif ONTOLOGY == "h21"
    ontology_index = 3
    CLASS_NUM = 480
elseif ONTOLOGY == "euler"
    ontology_index = 4
    CLASS_NUM = 961
end
if CLASS_NUM > 1
    CLASSES = collect(1:CLASS_NUM)
else
    # CLASSES = collect(-CLASS_NUM:2:960)
end
BATCH_SIZE = 1000
DATA_ID = "vector_data_with_features"

function __main__() 
    ############################
    # For startng a new data set
    ############################
    # new_sample(export_path="$PATH/$DATA_ID", split=true)

    @info("Importing Data...")
    train_set, test_set = import_split("$PATH/$DATA_ID/1/data");
    
    @info("Augmenting Data...")
    augment!(train_set)
    augment!(test_set)
    
    @info("Unpacking Data...")
    train_x, train_y = unpackage(train_set) 
    test_x, test_y =   unpackage(test_set)
    
    ######################################
    # Debugging and verification        
    sample = rand(1:size(train_set,1))    
    display(train_set[sample][1])         
    display(train_x[:,sample])            
    ######################################  

    if CLASS_NUM > 1
        if ONTOLOGY == "euler"
            @info("Mapping ontology to classification...")
            train_y = euler_to_index(train_y)
            test_y = euler_to_index(test_y)
        end
    end

    @info("Data loaded.")
    train_data = DataLoader(train_x, train_y, batchsize=BATCH_SIZE)
    global test_data = (test_x, test_y)

    @info("Building Model...")
    model_label="5 Hidden Layers w/ 1000 Neurons Per Layer"
    CLASS_NUM > 1 ? Output = softmax : Output = identity
    global model = Chain( 
                   Dense(MAX_SIZE*DIM,500,relu),
                   Dense(500,500,relu),
                   Dense(500,500,relu),
                   Dense(500,500,relu),
                   Dense(500,500,relu),
                   Dense(500,500,relu),
                   Dense(500,CLASS_NUM, relu),
                   Output);

    @info("Passing to GPU...")
    model |> gpu
    train_data |> gpu
    test_data |> gpu

    # sqnorm(x) = sum(abs2, x)
    opt = ADAM()
    acc(x,y) = sum(map(argmax, eachcol(model(x))) .== y)/size(y,1)
    if CLASS_NUM > 1
        loss(x,y) = Flux.crossentropy(model(x), Flux.onehotbatch(y, CLASSES)) #+ sum(sqnorm, Flux.params(model))
    else
        loss(x, y) = Flux.mse(model(x), Flux.onehotbatch(y,CLASSES))
    end

    Train(model, "$PATH/$DATA_ID", train_data, test_data, opt, loss, acc)

    model_dir = "$PATH/$DATA_ID/models/$ONTOLOGY/euler-2021-04-22T01:09:30.459_acc_28.6945.bson"
    eval_data = [test_x[:,i] for i ∈ rand(1:size(test_x,2), 10)]
    R = LRP(model_dir, eval_data)
    p_relevance = new_params(plot_heatmap; title="Relevance of input neurons", x_label=raw"\Chi_{\rm{actual}} - \Chi_{\rm{predicted}}", y_label="Input Neuron", fig_dir="/home/oppenheimer/Dev/calabiyau/figs/relevance.pdf")
    xs = string.([test_y[i] - argmax(model(eval_data[i])) for i=1:length(eval_data)])
    ys = string.(collect(1:136))
    heatmap(xs, ys, R, clim=(-5,5), dpi=300, c=cgrad([:white,:black,:white]))
end

@info("Headers compiled.")

# __main__()  
