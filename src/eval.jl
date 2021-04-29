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
        while length(v) > 4; pop!(v);end
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
    # ϵ = [0.0, 0.25, 0.0]
    # γ = [0.25, 0.0, 0.0]
    # lrp_layers = [floor((length(W) - 1) / 3) for i=1:3]
    # remainder = length(W) - sum(lrp_layers)
    # remainder > 0 && (lrp_layers[3] = lrp_layers[3] + remainder)
    # lrp_layers = cumsum(lrp_layers)
    # for m ∈ length(W):-1:1
    #     ms = m*ones(3)
    #     α = findall(x -> 0 ≤ x ≤ m + 1, lrp_layers - ms)
    #     for j ∈ 1:length(R[m])
    #         for k ∈ 1:length(R[m + 1])
    #             z = sum([a[m][l]*(W[m][k,l] + (γ[α] * relu(W[m][k,l]) )[1] ) for l ∈ 1:length(a[m])]) + b[m][k] + ϵ[α][1]
    #             s = R[m + 1][k] / z
    #             c1 = W[m][k,j] * s
    #             j == 1 && (R₀[m] += (b[m][k] * s))
    #             R[m][j] += a[m][j] * c1
    #         end
    #     end
    # end
    for m ∈ length(W):-1:1
        for j ∈ 1:length(R[m])
            for k ∈ 1:length(R[m + 1])
                z = sum([a[m][l]*W[m][k,l] for l ∈ 1:length(a[m])]) + b[m][k]
                s = R[m + 1][k] / z
                c1 = W[m][k,j] * s
                j == 1 && (R₀[m] += (b[m][k] * s))
                R[m][j] += a[m][j] * c1
            end
        end
    end
    return (R, R₀)
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
function LRP(model_dir, data; normalize=false)
    @load model_dir model
    P = params(model)
    W = [P[i] for i ∈ 1:2:length(P)]
    b = [P[i] for i ∈ 2:2:length(P)]
    R = zeros(length(data[1]), length(data))
    for (i,d) ∈ enumerate(data)
        r = relevance(W, b, d)
        R[:,i] = r[1][1]
    end
    if normalize == true
        R_max = map(maximum, eachcol(abs.(R)))
        for i = 1:10
            R[:,i] = abs.(R[:,i] / R_max[i])
        end
    end
    return R
end

function LRP(model::Chain, data; normalize=false)
    P = params(model)
    W = [P[i] for i ∈ 1:2:length(P)]
    b = [P[i] for i ∈ 2:2:length(P)]
    R = zeros(length(data[1]), length(data))
    R₀ = []
    for (i,d) ∈ enumerate(data)
        r = relevance(W, b, d)
        R[:,i] = r[1][1]
        push!(R₀ ,r[2])
    end
    if normalize == true
        R_max = map(maximum, eachcol(abs.(R)))
        for i = 1:10
            R[:,i] = abs.(R[:,i] / R_max[i])
        end
    end
    return R, R₀
end