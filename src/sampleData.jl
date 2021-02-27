using DataFrames, CSVFiles, CSV, CodecZlib, DelimitedFiles
using Random

# TODO: DOCUMENTATION
include("params.jl")

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
    return convert(Array, df[rowHead:rowFoot, colHead:colFoot] )
end

function ExportData(out, dir, file)
    open("$dir/$file", "w") do io
        @info("Opening ", "$dir/$file")
        stream = GzipCompressorStream(io)
        @info("Writing to path...")
        CSV.write(stream, out)
        close(stream)
        @info("Data exported.")
    end
end

function ExportCSV(csv::Array, dir::String, file_name::String)
    open("$dir/$file_name", "w") do f
        writedlm(f, csv, ",")
    end
end

function SplitAndExport(data, path; train_percent=0.8)
    train_set = []
    test_set = []
    shuffle!(data)
    @info("Splitting data...")
    for (i,d) in enumerate(data)
        if i/size(data,1) ≤ train_percent
            push!(train_set, d)
        else
            push!(test_set, d)
        end
    end
    n = 1
    if !isdir("$path/$n")
        try
            mkpath("$path/$n/data")
            mkpath("$path/$n/models")
        catch e
            @error("DATA NOT EXPORTED: ", e)
        end
    else
        while(isdir(path*"/$n"))
            n += 1
        end
        try
            mkpath("$path/$n/data")
        catch e
            @error("DATA NOT EXPORTED: ", e)
        end
    end
    export_path = "$path/$n/data"
    train_data = ArrayToDataFrame(train_set)
    test_data = ArrayToDataFrame(test_set)
    ExportData(train_data, export_path, "train.gz")
    ExportData(test_data, export_path, "test.gz")
end

function LoadDB(path; train_data=false, test_data=false)
    if train_data
        return DataFrame(load(File(format"CSV", path*"/train.gz")))
    elseif test_data
        return DataFrame(load(File(format"CSV", path*"/test.gz")))
    end
end

function LoadDB(path, index) 
    return DataFrame(load(File(format"CSV", path*"/$index.gz")));
end

function ImportData(import_path::String, dbId::Integer; train_data=false, test_data=false)
    df = LoadDB(import_path, train_data=train_data, test_data=test_data)
    data = DataFrameToArray(df, dbId=dbId, reimport=true)
    @info("IMPORTED ", size(df,1), " POLYTOPES.")
    return data
end

function ImportSplitData(import_path::String)
    train_df = LoadDB(import_path, train_data=true)
    test_df = LoadDB(import_path, test_data=true)
    train_set = DataFrameToArray(train_df, reimport=true)
    test_set = DataFrameToArray(test_df, reimport=true)
    return train_set, test_set  
end

function ArrayToDataFrame(data)
    out = DataFrame()
    @info("Exporting...")
    for d in data
        len = convert(Int32, size(d[1],1))
        delim = zeros(len)
        Δ = hcat(delim, d[1])
        header = zeros(DIM + 1)
        header[1] += len
        header[ontology_index] += d[2]
        Δ = vcat(header', Δ)
        Δ = convert(DataFrame, Δ)
        append!(out, Δ)
    end
    return out
end

function DataFrameToArray(df::DataFrame; dbId=nothing, required=[], reimport=false)
    if isnothing(dbId); dbId = 1; end
    dfPolyId = (dbId - 1) * SAMPLE_SIZE + 1
    data = []
    currentRow = 1    
    skip = GetDFValue(df, currentRow, 1, Int32)
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
            currentRow += skip + 1
            #iterate polytope count
            dfPolyId += 1
        end
        if currentRow + 1 > dfSize; break; end
        if currentRow + skip > dfSize; skip = GetDFValue(df, currentRow, 1, Int32); end
        #When we arrive at the correct polytope, grab the matrix and the ontology and add them to the dataset
        polyMat = GetDFMatrix(df, currentRow + 1, currentRow + skip, 2, DIM + 1)
        ontology = GetDFValue(df, currentRow, ontology_index, Int32)
        if reimport == false
            v = GetDFValue(df, currentRow + 1, 1, Int32)
            p = GetDFValue(df, currentRow + 2, 1, Int32)
            v_dual = GetDFValue(df, currentRow + 3, 1, Int32)
            p_dual = GetDFValue(df, currentRow + 4, 1, Int32)
            polyMat = vcat(polyMat, [v p v_dual p_dual])
        end
        push!(data, (polyMat,ontology))
    end
    println("Database: ", dbId) 
    println("REQUIRED: $(length(required))")
    println("GOT: $(length(data)) ", length(data) == length(required))
    println("---------------")
    # convert(Array{Tuple{Array{Float64,2},Int32}}, data)
    return data
end

function GetNewSample(;required = nothing, db_path = DB_PATH, export_path = "None", split = false, train_percent = 0.8, num_db::Integer = TOTAL_NUM_DB) 
    if required ≡ nothing
        required = RandomSample(SAMPLE_SIZE);
    end

    data = []
    for i in 1:num_db
        df = LoadDB(db_path, i);
        dbData = DataFrameToArray(df, dbId=i, required=required[i])
        append!(data, dbData)
    end
    @info("OBTAINED: $(size(data,1)) IN TOTAL")
    
    if export_path != "None"
        try
            if split == true 
                SplitAndExport(data, export_path; train_percent=train_percent)
                ExportCSV(required, "$export_path/$DATA_ID/data", "data_indices.csv")
            else
                n = 1; while(isdir("$export_path/$n.gz")) n+=1; end
                ExportData(data, export_path, "$n.gz")
            end
        catch error
            @error("DATA NOT EXPORTED: ", error)
        end
    else 
        @warn("DATA NOT EXPORTED: NO export_path GIVEN.")
    end
end
