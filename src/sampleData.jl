using DataFrames, CSV, FileIO, CodecZlib

const TOTAL_DATA = 473800746
const TOTAL_NUM_DB = 474
const DIM = 4
const sampleSize = 1_000_000

function LoadDB(index::Integer)
    df = DataFrame(load(File(format"CSV", "/media/share/Dev/CalabiYau/data/polytopes_db_4d/$index.gz")))
    return df
end

function RandomSample(sampleSize::Integer)
    binnedSample = []
    sample = rand(1:TOTAL_DATA, sampleSize)
    bins = [x for x in 0:sampleSize:TOTAL_DATA]
    push!(bins, TOTAL_DATA)

    sort!(sample)
    for i in 2:length(bins)
        subsample = []
        for (j,X) in enumerate(sample)

            if bins[i-1] < X ≤ bins[i]
                push!(subsample, X) 
            else 
                sample = sample[j:end]
                break 
            end
        end
        if !isempty(subsample) push!(binnedSample, subsample);end;
    end

    return convert(Array{Array{Int32}}, binnedSample)
end

function GetDFValue(df::DataFrame, row::Integer, col::Integer, T::Type)
    return convert(T, df[row, col])
end

function GetDFMatrix(df::DataFrame, rowHead::Integer, rowFoot::Integer, colHead::Integer, colFoot::Integer)
    return df[rowHead:rowFoot, colHead:colFoot]
end

function SearchDB(df::DataFrame, required::Array, dbId::Integer)
    sort!(required)
    println("Database: ", dbId, "\n REQUIRED: $(length(required))")

    data = []
    currentRow = 1;
    dfPolyId = (dbId - 1) * sampleSize;
    skip = 0;
    dfSize = size(df,1)
    for reqPolyId in required
        while(reqPolyId != dfPolyId && currentRow < dfSize)
            #get skip
            skip = GetDFValue(df, currentRow, 1, Int32)
            #increase currentRow
            currentRow += skip
            #iterate polytope count
            dfPolyId += 1
        end
        if currentRow + 1 > dfSize break; end
        #When we arrive at the correct polytope, grab the matrix and the ontology and add them to the dataset
        polyMat = GetDFMatrix(df, currentRow, currentRow + skip - 1, 2, DIM + 1)
        ontology = GetDFValue(df, currentRow + 1, 1, Int32)
        push!(data, (polyMat,ontology))
    end
    println("GOT: $(length(data))")
    println(length(data) == length(required))
    return convert(Array{Tuple{Array{Float64,2},Int32}}, data)
end

function ExportData(data, path)
    dataOut = DataFrame()
    
    for d in data
        len = convert(Int32, size(d[1],1))
        delim = zeros(len)
        delim[1] += len
        delim[2] += d[2]
        Δ = hcat(delim, d[1])
        Δ = convert(DataFrame, Δ)
        append!(dataOut, Δ)
    end
    
    open(path, "w") do io
        println("Opening ", path)
        stream = GzipCompressorStream(io)
        println("Writing to path...")
        CSV.write(stream, dataOut)
        close(stream)
        println("Data exported.")
    end
end

function GetDataSample(required::Array{Array{Int32}}; num_db::Integer = TOTAL_NUM_DB, export_path="None") 
    fullData = []
    for i in 1:num_db
        df = LoadDB(i);
        dbData = SearchDB(df, required[i], i)
        append!(fullData, dbData)
    end
    println("OBTAINED: $(size(fullData,1)) IN TOTAL")
    if export_path != "None" ExportData(fullData, export_path);else println("DATA NOT EXPORTED: NO export_path GIVEN."); end
    return fullData
end

function __main__()  
    required = RandomSample(sampleSize)
    fullData = GetDataSample(required, export_path="/home/oppenheimer/Dev/calabiyau/models/data/1.gz")
end

__main__()
