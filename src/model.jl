@info("Precompiling headers...")
using Flux, Zygote, CUDA, Random
using Flux:@epochs
using Flux.Data:DataLoader
using Flux:onehotbatch
using Printf, BSON, Dates, DelimitedFiles

include("data.jl")
include("train.jl")
include("eval.jl")

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
CLASS_NUM = 961
if CLASS_NUM > 1
    CLASSES = collect(1:CLASS_NUM)
else
    CLASSES = collect(-960:2:960)
end
BATCH_SIZE = 1000
ONTOLOGY = "euler"
DATA_ID = 1

if ONTOLOGY == "h11"
    ontology_index = 2
elseif ONTOLOGY == "h21"
    ontology_index = 3
elseif ONTOLOGY == "euler"
    ontology_index = 4
end

function __main__() 
    ############################
    # For startng a new data set
    ############################
    new_sample(export_path="$PATH/$ONTOLOGY", split=true)

    @info("Importing Data...")
    train_set, test_set = import_split("$PATH/$ONTOLOGY/$DATA_ID/data");
    
    # @info("Augmenting Data...")
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
        elseif ONTOLOGY == "h11" || ONTOLOGY == "h21"
            @info("Mapping ontology to classification...")
            train_y = hodge_to_index(train_y)
            test_y = hodge_to_index(test_y)
        end
    end

    # @info("Reshaping Data...")
    # train_x = reshape(train_x, MAX_SIZE, DIM, 1, size(train_x,3))
    # test_X = reshape(test_x, MAX_SIZE, DIM, 1, size(test_x,3))

    @info("Data loaded.")
    train_data = DataLoader(train_x, train_y, batchsize=BATCH_SIZE)
    global test_data = (test_x, test_y)

    @info("Building Model...")
    CLASS_NUM > 1 ? Output = softmax : Output = identity
    global model = Chain( 
                   #Conv((3,3), 1=>16, relu, pad=SamePad()),
                   #Conv((3,3), 16=>32, relu, pad=SamePad()),
                   #Conv((3,3), 32=>32, relu, pad=SamePad()),
                   #x -> x .= x[:,],
                   Dense(MAX_SIZE*DIM,356,elu),
                   Dense(356,356,relu),
                   Dense(356,356,relu),
                   Dense(356,356,relu),
                   Dense(356,CLASS_NUM, relu),
                   Output);

    @info("Passing to GPU...")
    model |> gpu
    train_data |> gpu
    test_data |> gpu

    # sqnorm(x) = sum(abs2, x)
    opt = ADAM()
    acc(x,y) = accuracy(x,y,model)
    if CLASS_NUM > 1
        loss(x,y) = Flux.crossentropy(model(x), Flux.onehotbatch(y, CLASSES)) #+ sum(sqnorm, Flux.params(model))
    else
        loss(x, y) = Flux.mse(model(x), Flux.onehotbatch(y,CLASSES))
    end
    Train(model, "$PATH/$ONTOLOGY/$DATA_ID", train_data, test_data, opt, loss, acc)
end

@info("Headers compiled.")

__main__()  
