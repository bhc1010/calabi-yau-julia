## Preformance Evaluation

# Confusion Matrix

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

function error_stats(gts, pred, err)
    error = err.(gts, pred)
    μ = round(mean(error), digits=6)
    σ = round(std(error, mean=μ), digits=6)
    return error, μ, σ
end