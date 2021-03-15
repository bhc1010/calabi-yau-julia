# Training

regression_acc(x,y,model) = sum(round.(model(x)) .== reshape(y,1,size(y,1)))/size(y,1);

classification_acc(x,y,model) = sum(map(argmax, eachcol(model(x))) .== y)/size(y,1);

accuracy(x,y,model) = CLASS_NUM > 1 ? classification_acc(x,y,model) : regression_acc(x,y,model)

function logging_train!(loss, ps, data, opt, losses)
    for (i,d) in enumerate(data)
        # back is a method that computes the product of the gradient so far with its argument.
        train_loss, back = Zygote.pullback(() -> loss(d...), ps)
        # Insert whatever code you want here that needs training_loss, e.g. logging.
        # Apply back() to the correct type of 1.0 to get the gradient of loss.
        gs = back(one(train_loss))
        push!(losses, train_loss)
        # Insert what ever code you want here that needs gradient.
        # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
        Flux.update!(opt, ps, gs)
        # Here you might like to check validation set accuracy, and break out to do early stopping.
    end
end

function Train(model, dir,train_data, test_data, opt, loss, accuracy; epochs=100::Integer)
    ps = Flux.params(model)
    best_acc = 0.0
    # best_loss = 9999999
    last_improvement = 0
    test_x = test_data[1]
    test_y = test_data[2]
    model_name = ""
    dir = "$dir/models"
    loss_log = []
    @info("Beginning training loop...")
    for epoch_idx in 1:epochs
        losses = []
        # Train for a single epoch
        # Flux.train!(loss, ps, train_data, opt)
        logging_train!(loss, ps, train_data, opt, losses)
        # Calculate loss
        last_loss = losses[end]
        append!(loss_log, last_loss)
        # Calculate accuracy:
        acc = accuracy(test_x, test_y)
        @info(@sprintf("[%d]: Validation loss: %.6f", epoch_idx, last_loss))
        @info(@sprintf("[%d]: Validation accuracy: %f", epoch_idx, acc))
        
        # If our accuracy is good enough, quit out.
        if acc >= 0.999
            @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
            break
        end
        
        if acc > best_acc
            model_name = "model-$(now())-CLASS_NUM-$CLASS_NUM-acc-$(acc*100).bson"
            @info(" -> New best loss! Saving model out to $dir/$model_name")
            if !ispath(dir)
                mkpath(dir)
            else
                rm(dir, recursive=true)
                mkpath(dir)
            end
            BSON.@save "$dir/$model_name" model epoch_idx opt acc
            last_improvement = epoch_idx
            best_acc = acc
        end

        # If we haven't seen improvement in 20 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
            opt.eta /= 10.0
            @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
            
            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end
        
        if epoch_idx - last_improvement >= 10
            @warn("Accuracy no longer increasing. Exiting early.")
            break
        end
    end
    ## Export loss log for plotting 
    export_csv(losses, "$dir/logs", "$model_name.csv")
end