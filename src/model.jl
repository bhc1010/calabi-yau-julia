@info("Precompiling headers...")
using Flux, Zygote, CUDA, Random
using Flux:@epochs
using Flux.Data:DataLoader
using Flux:onehotbatch
using Printf, BSON, Dates, DelimitedFiles

include("sampleData.jl")

const MAX_SIZE = 34
const CLASS_NUM = 961
const CLASSES = [x for x in 1:CLASS_NUM]
const BATCH_SIZE = 1000
const MODEL_NAME = Dates.format(now(), "mm-dd-yy-HH:MM:SS")*"-$CLASS_NUM"
const ONTOLOGY = "euler"
losses = []

accR(x,y,model) = sum(round.(model(x)) .== y)/size(y,1);
accC(x,y,model) = sum(map(argmax, eachcol(model(x))) .== y)/size(y,1);
accuracy(x,y,model) = CLASS_NUM > 1 ? accC(x,y,model) : accR(x,y,model)
EulerToIndex(y) = convert.(Int32, map(x->0.5*(x+960)+1, y))
HodgeToIndex(y) = convert.(Int32, map(x->0.5*(x+450)+1,y))
SetOntology(ontology::String) = ONTOLOGY = ontology;

function AugmentData!(data)
    X = zeros(MAX_SIZE,DIM,size(data,1))
    Y = []
    augmentedX = []

    for d in data
        diff = MAX_SIZE - size(d[1],1)
        push!(augmentedX, [d[1];zeros(diff,DIM)])
        push!(Y, d[2])
    end

    for (i,d) in enumerate(augmentedX)
        X[:,:,i] = X[:,:,i] + d
    end

    for i in 1:length(data)
        data[i] = (X[:,:,i], Y[i])
    end
end

function UnpackageData(data)
    X = zeros(MAX_SIZE, DIM, size(data,1))
    Y = []
    for (i,d) in enumerate(data)
        X[:,:,i] = X[:,:,i] + d[1]
        append!(Y, d[2])
    end
    return X, Y
end

function logging_train!(loss, ps, data, opt)
    global losses
    for d in data
      # back is a method that computes the product of the gradient so far with its argument.
      train_loss, back = Zygote.pullback(() -> loss(d...), ps)
      # Insert whatever code you want here that needs training_loss, e.g. logging.
      push!(losses, train_loss[end])
      # Apply back() to the correct type of 1.0 to get the gradient of loss.
      gs = back(one(train_loss))
      # Insert what ever code you want here that needs gradient.
      # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
      Flux.update!(opt, ps, gs)
      # Here you might like to check validation set accuracy, and break out to do early stopping.
    end
end

function Train(model, train_data, test_data, opt, loss, accuracy; epochs=100::Integer)
    ps = Flux.params(model)
    best_acc = 0.0
    best_loss = 9999999
    last_improvement = 0
    test_x = test_data[1]
    test_y = test_data[2]
    dir = "/home/oppenheimer/Dev/calabiyau/trained/models"
    n=0
    while(isdir(dir*"/$n")); n += 1; end
    dir = dir*"/$n"
    loss_log = dir*"/log.txt"
    global losses

    @info("Beginning training loop...")
    for epoch_idx in 1:epochs
        # Train for a single epoch
        # Flux.train!(loss, ps, train_data, opt)
        logging_train!(loss, ps, train_data, opt)
        last_loss = losses[end]
        # Calculate accuracy:
        acc = accuracy(test_x,test_y)
        @info(@sprintf("[%d]: Validation loss: %.6f", epoch_idx, last_loss))
        @info(@sprintf("[%d]: Validation accuracy: %.6f", epoch_idx, acc))
        
        # If our accuracy is good enough, quit out.
        if acc >= 0.999
            @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
            break
        end
        
        if last_loss < best_loss
            @info(" -> New best loss! Saving model out to $dir/model-$(now())-CLASS_NUM-$CLASS_NUM-acc-$(acc*100).bson")
            if !isdir(dir)
                mkdir(dir)
            else
                rm(dir, recursive=true)
                mkdir(dir)
            end
            BSON.@save "$dir/model-$(now())-CLASS_NUM-$CLASS_NUM-acc-$(acc*100).bson" model_dir model epoch_idx opt acc
            last_improvement = epoch_idx
            best_loss = last_loss
        end

        # If we haven't seen improvement in 20 epochs, drop our learning rate:
        # if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
        #     opt.eta /= 10.0
        #     @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
            
        #     # After dropping learning rate, give it a few epochs to improve
        #     last_improvement = epoch_idx
        # end
        
        if epoch_idx - last_improvement >= 10
            @warn("Loss no longer decreasing. Exiting early.")
            break
        end
    end
    open(loss_log, "w") do file
        writedlm(file, losses, delim=',')
    end
end

function __main__() 

    GetNewSample(export_path=PATH, split=true, train_percent=0.8, ontology=ONTOLOGY)

    @info("Importing Data...")
    train_set, test_set = ImportSplitData("/home/oppenheimer/Dev/calabiyau/trained/$EXPORT_PATH/data", export_csv=true)
    
    @info("Augmenting Data...")
    AugmentData!(train_set)
    AugmentData!(test_set)

    @info("Unpacking Data...")
    train_x, train_y = UnpackageData(train_set)
    test_x, test_y = UnpackageData(test_set)

    if ONTOLOGY == "euler"
        @info("Mapping ontology to classification...")
        train_y = EulerToIndex(train_y)
        test_y = EulerToIndex(test_y)
    elseif ONTOLOGY == "h11" || ONTOLOGY == "h21"
        @info("Mapping ontology to classification...")
        train_y = HodgeToIndex(train_y)
        test_y = HodgeToIndex(test_y)
    end
    @info("Reshaping Data...")
    train_x = reshape(train_x, MAX_SIZE, DIM, 1, size(train_x,3))
    test_X = reshape(test_x, MAX_SIZE, DIM, 1, size(test_x,3))

    @info("Data loaded.")
    train_data = DataLoader(train_x, train_y, batchsize=BATCH_SIZE);
    test_data = (test_x, test_y)

    @info("Building Model...")
    CLASS_NUM > 1 ? Output = softmax : Output = identity
    model = Chain( 
                #    Conv((3,3), 1=>32, relu, pad=4),
                #    Conv((3,3), 32=>64, relu, pad=4),
                #    Conv((3,3), 64=>32, relu, pad=4),
                   Flux.flatten,
                   Dense(136,500,relu),
                   Dense(500,500,relu),
                   Dense(500,500,relu),
                   Dense(500,500,relu),
                   Dense(500,500,relu),
                   Dense(500,500,relu),
                   Dense(500,CLASS_NUM,relu),
                   Output);

    @info("Passing to GPU...")
    model |> gpu
    train_data |> gpu
    test_data |> gpu

    opt = ADAM()
    loss(x, y) = Flux.mse(model(x), Flux.onehotbatch(y,CLASSES))
    acc(x,y) = accuracy(x,y,model)
    Train(model, train_data, test_data, opt, loss, acc)
end

@info("Headers compiled.")

__main__()
