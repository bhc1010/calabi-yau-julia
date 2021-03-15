using CSVFiles, CSV, DelimitedFiles
using DataFrames
using CodecZlib
using Random

flatten(x::Array{<:Array,1})= Iterators.flatten(x)|> collect|> flatten
flatten(x::Array{<:Number,1})= x

function random_sample(sample_size::Integer = SAMPLE_SIZE)
    binned_sample = []
    sample = rand(1:TOTAL_DATA, sample_size)
    bins = push!([x for x in 0:sample_size:TOTAL_DATA], TOTAL_DATA)
    
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
        !isempty(subsample) && push!(binned_sample, subsample)
    end
    
    return convert(Array{Array{Int32}}, binned_sample)
end

function viewdf(df::DataFrame, row, col, T::Type)
    return convert(T, df[row, col])
end

function export_gz(out, dir, file)
    open("$dir/$file", "w") do io
        @info("Opening ", "$dir/$file")
        stream = GzipCompressorStream(io)
        @info("Writing to path...")
        CSV.write(stream, out)
        close(stream)
        @info("Data exported.")
    end
end

function export_csv(csv, dir::String, file_name::String)
    open("$dir/$file_name", "w") do f
        writedlm(f, csv, ',')
    end
end

function permute!(data)
    Π = randperm(size(data,1))
    data[:] = data[Π]
    return Π
end

function split(data, train_percent)
    train_data = []
    test_data = []
    for (i,d) in enumerate(data)
        if i/size(data,1) ≤ train_percent
            push!(train_data, d)
        else
            push!(test_data, d)
        end
    end
    return train_data, test_data
end

"""
    _enum_mkpaths(path, subdirs)

Makes paths at path/n/subdirs[1], ... ,path/n/subdirs[end]
where n is the largest integer directory name contained in path.

Returns the first subdirectory path (/path/n/subdirs[1]).
"""
function _enum_mkpaths(path::String, subdirs::AbstractArray{<:String})
    n = 1
    while(isdir("$path/$n"))
        n += 1
    end
    
    [mkpath("$path/$n/$sd") for sd in subdirs]

    return "$path/$n/$(subdirs[1])"
end

function split_export(data, path; train_percent=0.8, csv=nothing)
    
    Π = permute!(data)
    train_set, test_set = split(data, train_percent)
    export_path = _enum_mkpaths(path, ["data", "models", "models/logs"])

    train_data = array_to_df(train_set)
    test_data = array_to_df(test_set)
    export_gz(train_data, export_path, "train.gz")
    export_gz(test_data, export_path, "test.gz")

    if !(csv ≡ nothing)  
        csv = flatten(csv)[Π]
        train_idx, test_idx = split(csv, train_percent)
        export_csv(train_idx, "$export_path", "train_idx.csv")
        export_csv(test_idx, "$export_path", "test_idx.csv")
    end
end

function loaddf(path; train_data=false, test_data=false)
    if train_data
        return DataFrame(load(File(format"CSV", path*"/train.gz")))
    elseif test_data
        return DataFrame(load(File(format"CSV", path*"/test.gz")))
    end
end

loaddf(path, index) = DataFrame(load(File(format"CSV", "$path/$index.gz")));

function import_data(import_path::String, dbId::Integer; train_data=false, test_data=false)
    df = loaddf(import_path, train_data=train_data, test_data=test_data)
    data = df_to_array(df, dbId=dbId, reimport=true)
    @info("IMPORTED ", size(df,1), " POLYTOPES.")
    return data
end

function import_split(import_path::String)
    train_df = loaddf(import_path, train_data=true)
    test_df = loaddf(import_path, test_data=true)
    train_set = df_to_array(train_df, reimport=true)
    test_set = df_to_array(test_df, reimport=true)
    return train_set, test_set
end

function array_to_df(data)
    out = DataFrame()
    @info("Exporting...")
    for d in data
        ## Need to transpose matrix again for columnar output
        len = convert(Int32, size(d[1]',1))
        delim = zeros(len)
        Δ = hcat(delim, d[1]')
        header = zeros(DIM + 1)
        header[1] += len
        header[ontology_index] += d[2]
        Δ = vcat(header', Δ)
        Δ = convert(DataFrame, Δ)
        append!(out, Δ)
    end
    return out
end

function df_to_array(df::DataFrame; dbId=nothing, required=nothing, reimport=false)
    dbId ≡ nothing && (dbId = 1)
    dfPolyId = (dbId - 1) * SAMPLE_SIZE + 1
    data = []
    currentRow = 1    
    skip = viewdf(df, currentRow, 1, Int32)
    dfSize = size(df,1)
    polyMat = []

    if reimport == true
        required = [x for x in 1:SAMPLE_SIZE]
        dfPolyId = 1
    elseif required ≡ nothing
        required = [x for x in ((dbId - 1) * SAMPLE_SIZE + 1 ):(dbId * SAMPLE_SIZE)]
    end

    for reqPolyId in required
        while(reqPolyId != dfPolyId && currentRow < dfSize)
            #get skip
            skip = viewdf(df, currentRow, 1, Int32)
            #increase currentRow
            currentRow += skip + 1
            #iterate polytope count
            dfPolyId += 1
        end
        if currentRow + 1 > dfSize; break; end
        #When we arrive at the correct polytope, grab the matrix and the ontology and add them to the dataset
        skip = viewdf(df, currentRow, 1, Int32)
        NUMBERS_ONLY == false && (polyMat = viewdf(df, currentRow+1:currentRow+skip, 2:DIM+1, Array))
        ontology = viewdf(df, currentRow, ontology_index, Int32)
        ############################ 
        #Debugging and verification
        ############################
        # if reqPolyId == 100000
        #     println(currentRow)
        #     display(df[currentRow:currentRow+skip+1,:])
        # end
        ###########################
        
        if reimport == false
            if NUMBERS_ONLY == false
                v = viewdf(df, currentRow + 1, 1, Int32)
                p = viewdf(df, currentRow + 2, 1, Int32)
                v_dual = viewdf(df, currentRow + 3, 1, Int32)
                p_dual = viewdf(df, currentRow + 4, 1, Int32)
                polyMat = vcat(polyMat, [v p v_dual p_dual])
            else
                polyMat = viewdf(df, currentRow + 1:currentRow + 4, 1:1, Array)'
            end
        else 
            NUMBERS_ONLY == true && (polyMat = viewdf(df, currentRow + 1:currentRow + 1, 2:5, Array))
        end
        push!(data, (polyMat',ontology))
    end
    println("Database: ", dbId) 
    println("REQUIRED: $(length(required))")
    println("GOT: $(length(data)) ", length(data) == length(required))
    println("---------------")
    # convert(Array{Tuple{Array{Float64,2},Int32}}, data)
    return data
end

function new_sample(;required = nothing, db_path = DB_PATH,
                     export_path = "None",
                     sample_size = SAMPLE_SIZE,
                     split = false,
                     train_percent = 0.8,
                     db_start = 1,
                     db_end = TOTAL_NUM_DB,
                     get_all=false) 

    if required ≡ nothing
        required = random_sample(sample_size);
    end
    get_all == true && (required = [x for x in 1:sample_size])

    data = []
    for i in db_start:db_end
        df = loaddf(db_path, i);
        dbData = df_to_array(df, dbId=i, required=required[i])
        append!(data, dbData)
    end
    @info("OBTAINED: $(size(data,1)) IN TOTAL")
    
    if export_path != "None"
        if split == true 
            split_export(data, export_path; train_percent=train_percent, csv=required)
        else
            n = 1; while(isdir("$export_path/$n.gz")) n+=1; end
            export_gz(data, export_path, "$n.gz")
        end
    else 
        @warn("DATA NOT EXPORTED: NO export_path GIVEN.")
    end
end

function augment!(data)
    X = zeros(DIM,MAX_SIZE,size(data,1))
    Y = []
    augmentedX = []

    for d in data
        diff = MAX_SIZE - size(d[1],2)
        diff > 0 ? push!(augmentedX, [d[1] zeros(DIM, diff)]) : nothing
        push!(Y, d[2])
    end

    for (i,d) in enumerate(augmentedX)
        X[:,:,i] = X[:,:,i] + d
    end

    for i in 1:length(data)
        data[i] = (X[:,:,i], Y[i])
    end
end

function unpackage(data)
    X = zeros(DIM*MAX_SIZE, size(data,1))
    Y = []
    for (i,d) in enumerate(data)
        X[:,i] = X[:,i] + d[1][:,]
        append!(Y, d[2])
    end

    return X, Y
end

euler_to_index(x) = convert.(Int32, round.(map(y -> 0.5*(y + 960) + 1, x)))
index_to_euler(x) = convert.(Int32, round.(map(y -> 2*(y - 1) - 960, x)))
hodge_to_index(x) = convert.(Int32, map(y -> 0.5*(y + 450) + 1,x))
index_to_hodge(x) = convert.(Int32, map(y -> 2*(y - 1) - 450 ,x))