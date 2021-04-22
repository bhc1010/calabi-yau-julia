## Preformance Evaluation
using BSON: @load

"""
confusion_matrix(k::Integer, gts, preds):
Generates confusion matrix between gts and pred.
"""
function confusion_matrix(k::Integer, gts, preds)
    n = length(gts)
    length(preds) == n || throw(DimensionMismatch("Inconsistent lengths."))
    R = zeros(Int, k, k)
    for i = 1:n
        @inbounds g = gts[i]
        @inbounds p = preds[i]
        R[g, p] += 1
    end
    return R
end

"""
function error_stats(gts, pred, err):
Calculates error for each data point and overall mean error and std. Type of error is specified by function passed to err.
"""
function error_stats(gts, pred, err)
    error = err.(gts, pred)
    μ = round(mean(error), digits=6)
    σ = round(std(error), digits=6)
    return error, μ, σ
end

"""
rel_error(gts,pred):
Returns the relative error between gts and pred
"""
rel_error(gts,pred) = abs((gts-pred)/gts)

"""
calculate_leadingpiece(X):
Calculates leading order term of X. Depends on global parameter ONTOLOGY.
"""
function calculate_leadingpiece(X)
    leading = []
    for i ∈ 1:size(X,2)
        v = X[:,i]
        for x ∈ v[collect(length(v):-1:1)]
            (x == 0.0) && pop!(v)
            (x != 0.0) && break
        end
        while length(v) > 4; popfirst!(v);end
        pts = v[2]
        dualpts = v[4]
        if ONTOLOGY == "h11"
            push!(leading, pts - (DIM + 1))
        elseif ONTOLOGY == "h21"
            push!(leading, dualpts - (DIM + 1))
        else ONTOLOGY == "euler"
            push!(leading, 2*(pts - dualpts))
        end
    end
    return leading
end

"""
relevance(W, b, X):
Outputs vector R containing neuron-by-neuron relevance for each layer.
"""
function relevance(W, b, X)
    a = get_activations(W, b, X, relu)
    R = [zeros(length(a[i])) for i ∈ 1:length(a)]
    R₀ = [0.0 for i ∈ 1:length(a)]
    R[end][argmax(softmax(a[end]))] = 1
    ϵ = 1e-9
    for m ∈ length(W):-1:1
        for j ∈ 1:length(R[m])
            for k ∈ 1:length(R[m + 1])
                z = sum([a[m][l]*W[m][k,l] for l ∈ 1:length(a[m])]) + b[m][k] + ϵ
                s = R[m + 1][k] / z
                c1 = W[m][k,j] * s
                c2 = b[m][k] * s 
                R[m][j] += a[m][j] * c1 + c2/length(R[m])
            end
        end
    end
    return R
    
    # for m ∈ length(W):-1:1
    #     for j ∈ 1:length(R[m])
    #         for k ∈ 1:length(R[m + 1])
    #             z = sum([a[m][l]*W[m][k,l] for l ∈ 1:length(a[m])]) + b[m][k]
    #             s = R[m + 1][k] / z
    #             c1 = W[m][k,j] * s
    #             j == 1 && (R₀[m] += (b[m][k] * s))
    #             R[m][j] += a[m][j] * c1 #+ R₀[m]#+ c2/length(R[m])
    #         end
    #     end
    # end
    # return (R, R₀)
    
    # z = W[m]*a[m] + b[m] .+ ϵ
    # s = R[m + 1] ./ z
    # c = [sum([W[m][k,j] * s[k] for k ∈ 1:length(s)]) for j ∈ 1:size(W[m],2)]
    # R[m] = a[m] .* c
end

"""
get_activations(W, b, X, f):
Feeds X forward through the model defined by (W, b) with activation function f. 
Returns list a containing the values of each neuron for every layer.
"""
function get_activations(W, b, X, f)
    a = [X]
    for m ∈ 1:length(W)
        push!(a, f.(W[m]*a[m] + b[m]))
    end
    return a
end

"""
Layer-wise Relevance Propogation.
LRP(model_dir, data) : (model, X) ⟶ R = [r₁, … ,rₙ] where rᵢ is the relevance of layer i
"""
function LRP(model_dir, data)
    @load model_dir model
    P = params(model)
    W = [P[i] for i ∈ 1:2:length(P)]
    b = [P[i] for i ∈ 2:2:length(P)]
    R = zeros(length(data[1]), length(data))
    for (i,d) ∈ enumerate(data)
        r = relevance(W, b, d)
        R[:,i] = r[1]
    end
    return R
end