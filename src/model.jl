
using Flux, CUDA, Random, SparseArrays
using Flux:@epochs
using Flux.Data:DataLoader
using Printf, BSON, Dates

import("sampleData.jl")

const DIM = 4
const MAX_SIZE = 33
const MODE = "classification"
const CLASS_NUM = 961
const BATCH_SIZE = 1000
const MODEL_NAME = Times(Dates.now())*"-"*MODE

function AugmentData!(data)
    dataSize = size(data,1)
    X = zeros(MAX_SIZE,DIM,dataSize)
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

    X = reshape(X, MAX_SIZE, DIM, 1, dataSize);
    for i in 1:length(data)
        data[i] = (X[:,:,1,i], Y[i])
    end
end

onehot_neg(y) = sparsevec.([y + CLASS_NUM], [1]) 

function accuracy(x,y,model)
    MODE == "regression" ? accR(x,y,model) : nothing
    MODE == "classification" ? accC(x,y,model) : nothing
end

accR(x,y,model) = sum(round.(model(x)) .== y)/size(y,1)/100;
accC(x,y,model) = sum(findmax.(model(x)) .== onehot_neg(y))/size(y,1)/100;

function Train(model, train_data, test_data, opt, loss, accuracy; epochs=100::Integer)
    ps = Flux.params(model)
    best_acc = 0.0
    last_improvement = 0
    test_x = test_data[1]
    test_y = test_data[2]

    @info("Beginning training loop...")
    for epoch_idx in 1:epochs
        # Train for a single epoch
        Flux.train!(loss, ps, train_data, opt)

        # Calculate accuracy:
        acc = accuracy(test_x,test_y, model)
        @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
        
    #     If our accuracy is good enough, quit out.
        # if acc >= 0.999
        #     @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        #     break
        # end

        # If this is the best accuracy we've seen so far, save the model out
        if acc >= best_acc
            model_name = MODEL_NAME*"-"*acc
            @info(" -> New best accuracy! Saving model out to $model_name.bson")
            BSON.@save model_name model epoch_idx acc
            best_acc = acc
            last_improvement = epoch_idx
        end

        # If we haven't seen improvement in 20 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
            opt.eta /= 10.0
            @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end

        if epoch_idx - last_improvement >= 10
            @warn(" -> We're calling this converged.")
            break
        end
    end
end

function __main__()  

    train_set, test_set = shuffle!(ImportSplitData("/home/oppenheimer/Dev/calabiyau/models/data", 1, catch_errors=true))
    
    AugmentData!(train_set)
    AugmentData!(test_set)

    train_x = [x[1] for x in train_set]
    train_y = [x[2] for x in train_set]
    test_x = [x[1] for x in test_set] 
    test_y = [x[2] for x in test_set]

    train_data = DataLoader(train_x, train_y, batchsize=BATCH_SIZE);
    test_data = (test_x, test_y)

    MODE == "regression" ? outSize = 1 : outSize = CLASS_NUM
    MODE == "classification" ? Output = softmax : Output = identity

    model = Chain( Flux.flatten,
                   Dense(132,500,relu),
                   Dense(500,500,relu),
                   Dense(500,500,relu),
                   Dense(500,500,relu),
                   Dense(500,500,relu),
                   Dense(500,500, σ),
                   Dense(500,outSize),
                   Output);

    model |> gpu
    train_data |> gpu
    test_x |> gpu
    test_y |> gpu

    opt = ADAM()
    loss(x, y) = Flux.Losses.mse(model(x), y)
    acc == (x,y) -> accuracy(x,y)
    Train(model, train_data, test_data, opt, loss, acc)
end

__main__()