using DataFrames, CSV, FileIO, CodecZlib

const TOTAL_DATA = 473_800_746
const TOTAL_NUM_DB = 474
const DIM = 4
SAMPLE_SIZE = 1_000_000


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

SetSampleSize!(sampleSize::Integer) = global SAMPLE_SIZE = sampleSize;

SetPopulationSize!(popSize::Integer) = global TOTAL_DATA = popSize;

function DataFrameToArray(df::DataFrame, dbId::Integer;required=[], reimport=false)
    dfPolyId = (dbId - 1) * SAMPLE_SIZE + 1
    data = []
    currentRow = 1;
    skip = 0;
    dfSize = size(df,1)
    if reimport == true
        required = [x for x in 1:SAMPLE_SIZE]
        dfPolyId = 1
    elseif isempty(required)
        required = [x for x in ((dbId - 1) * SAMPLE_SIZE + 1 ):(dbId * SAMPLE_SIZE)]
    end
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
    println("Database: ", dbId) 
    println("REQUIRED: $(length(required))")
    println("GOT: $(length(data)) ", length(data) == length(required))
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
        @info("Opening ", path)
        stream = GzipCompressorStream(io)
        @info("Writing to path...")
        CSV.write(stream, dataOut)
        close(stream)
        @info("Data exported.")
    end
end

function SplitAndExport(data, id, path; train_percent=0.8)
    train_set = []
    test_set = []
    @info("Splitting data...")
    for (i,d) in enumerate(data)
        if i/size(data,1) < train_percent
            push!(train_set, d)
        else
            push!(test_set, d)
        end
    end
    n = 0
    while(!isdir(path*"/$n"))
        n += 1
    end
    try
        mkdir(path*"/$n")
    catch e
        throw(e)
        @warn("DATA NOT EXPORTED: NO DIRECTORY CREATED.")
    end
    path = path*"/$n"
    ExportData(train_set, path*"/train.gz")
    ExportData(test_set, path*"/test.gz")
end

function GetNewSample(;required=nothing, db_path="/media/share/Dev/CalabiYau/data/polytopes_db_4d", export_path="None", split=true, train_percent=0.8,num_db::Integer = TOTAL_NUM_DB) 
    if required ≡ nothing required = RandomSample(SAMPLE_SIZE); end;
    
    fullData = []
    for i in 1:num_db
        df = LoadDB(db_path, i, catch_errors=true);
        dbData = DataFrameToArray(df, i, required=required[i])
        append!(fullData, dbData)
    end
    @info("OBTAINED: $(size(fullData,1)) IN TOTAL")
    
    if export_path != "None" 
        try
            split == true ? SplitAndExport(fullData, export_path, train_percent=train_percent) : ExportData(fullData, export_path)
        catch error
            throw(error)
            @error("DATA NOT EXPORTED.")
        end
    else 
        @warn("DATA NOT EXPORTED: NO export_path GIVEN.")
    end
    return fullData
end

function LoadDB(path, index::Integer; train_data=false, test_data=false, catch_errors=false)
    if catch_errors == true
        try
            FIRST_IMPORT = DataFrame(load(File(format"CSV", "/home/oppenheimer/Dev/calabiyau/models/data/1.gz")))
        catch
            @info "Precompiling packages, ignoring errors..."
        end
    end
    
    if train_data
        return DataFrame(load(File(format"CSV", path*"/$index/train.gz")))
    elseif test_data
        return DataFrame(load(File(format"CSV", path*"/$index/test.gz")))
    else
        return DataFrame(load(File(format"CSV", path*"/$index.gz")))
    end
end

function ImportData(import_path::String, dbId::Integer; train_data=false, test_data=false, catch_errors=false)
    df = LoadDB(import_path, dbId, train_data=train_data, test_data=test_data ,catch_errors=catch_errors)
    data = DataFrameToArray(df, dbId, reimport=true)
    @info("IMPORTED ", size(df,1), " POLYTOPES.")
    return data
end

function ImportSplitData(import_path::String, dbId::Integer; catch_errors=false)
    train_df = LoadDB(import_path, dbId, train_data=true, catch_errors=catch_errors)
    test_df = LoadDB(import_path, dbId, test_data=true)
    train_data = DataFrameToArray(train_df, dbId, reimport=true)
    test_data = DataFrameToArray(test_df, dbId, reimport=true)
    @info("IMPORTED ", size(train_data,1), " POLYTOPES FOR TRAINING.")
    @info("IMPORTED ", size(test_data,1), " POLYTOPES FOR TESTING.")
    return train_data, test_data
end
