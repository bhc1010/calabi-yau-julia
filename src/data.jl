using CSVFiles, CSV, DelimitedFiles
using DataFrames
using CodecZlib
using Random

flatten(x::Array{<:Array,1})= Iterators.flatten(x)|> collect|> flatten
flatten(x::Array{<:Number,1})= x

"""
random_sample(sample_size::Integer = SAMPLE_SIZE)
Generates a random sample of indices from 1 to TOTAL_DATA and bins them with a bin width of sample_size.
"""
function random_sample(sample_size::Integer = SAMPLE_SIZE)
    sample = sort!(rand(1:TOTAL_DATA, sample_size))
    binned = bin(sample, TOTAL_DATA, sample_size)
    return convert(Array{Array{Int32,1}}, binned)
end

"""
bin(data, n, Δ)
Returns data binned into n bins of width Δ
"""
function bin(data, n, Δ)
    bins = [[] for i = 1:Δ:n]
    for i ∈ eachindex(bins)
        for (j, X) ∈ enumerate(data)
            if (i - 1)*Δ < X ≤ i*Δ
                push!(bins[i], X)
            else
                data = data[j:end]
                break
            end
        end
    end
    return bins
end

"""
viewdf(df::DataFrame, row, col, T::Type)
Returns the dataframe value at the specified row and column and converts this value to type T 
"""
function viewdf(df::DataFrame, row, col, T::Type)
    return convert(T, df[row, col])
end

"""
export_gz(out, dir, file)
Exports out argument as a .gzip file to "$dir/$file" 
"""
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

"""
export_csv(csv, dir::String, file_name::String)
Exports csv argument as a csv file to "$dir/$file"
"""
function export_csv(csv, dir::String, file::String)
    open("$dir/$file", "w") do f
        writedlm(f, csv, ',')
    end
end

"""
permute!(data)
Generates a permutation Π, permutes data by Π, and returns Π
"""
function permute!(data)
    Π = randperm(size(data,1))
    data[:] = data[Π]
    return Π
end

"""
split(data, train_percent)
Returns data split into two lists, the first containing the first $(train_percent) elements of data, the second containing the last $(1 - train_percent) elements of data.
"""
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
_mkpaths(path::String, subdirs::AbstractArray{<:String})

Makes paths at path/subdirs[1], ... ,path/subdirs[end]

Returns the first subdirectory path (/path/subdirs[1])
"""
function _mkpaths(path::String, subdirs::AbstractArray{<:String})
    [mkpath("$path/$sd") for sd in subdirs]
    return "$path/$n/$(subdirs[1])"
end

"""
split_export(data, onts, path; train_percent=0.8, csv=nothing)

Splits data by an amount $(train_percent) and exports both lists to a .gzip file. 

If csv is given, exports csv as .gzip file.
"""
function split_export(data, onts, path; train_percent=0.8, csv=nothing)
    total_data =[(d[1],o) for (d,o) ∈ zip(data, onts)]
    Π = permute!(total_data)
    train_set, test_set = split(total_data, train_percent)
    export_path = _mkpaths(path, ["data", "models/$ONTOLOGY/temp", "models/$ONTOLOGY/logs"])

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

"""
loaddf(path; train_data=false, test_data=false)
Loads training or testing dataframe
"""
function loaddf(path; train_data=false, test_data=false)
    if train_data
        return DataFrame(load(File(format"CSV", path*"/train.gz")))
    elseif test_data
        return DataFrame(load(File(format"CSV", path*"/test.gz")))
    end
end

"""
Loads dataframe specified by index
"""
loaddf(path, index) = DataFrame(load(File(format"CSV", "$path/$index.gz")));

"""
function import_csv(import_path::String; file = nothing)
Imports csv file at "$import_path/$file"
"""
function import_csv(import_path::String; file = nothing)
    csv = AbstractArray{Int}
    !isnothing(file) && (import_path = "$import_path/$file")
    open(import_path) do f
        csv = readdlm(import_path, ',', Int32)
    end
    return csv
end

"""
import_data(import_path::String, dbId::Integer; train_data=false, test_data=false)
Returns data from dataframe at import_path with database id dbID
"""
function import_data(import_path::String, dbId::Integer; train_data=false, test_data=false)
    df = loaddf(import_path, train_data=train_data, test_data=test_data)
    data = df_to_array(df, dbId=dbId, reimport=true)
    @info("IMPORTED ", size(df,1), " POLYTOPES.")
    return data
end

"""
import_split(import_path::String)
Returns training and testing data after importing from import_path
"""
function import_split(import_path::String)
    train_df = loaddf(import_path, train_data=true)
    test_df = loaddf(import_path, test_data=true)
    train_set = df_to_array(train_df, reimport=true)
    test_set = df_to_array(test_df, reimport=true)
    return train_set, test_set
end

"""
array_to_df(data)
Converts array data to dataframe for exporting
"""
function array_to_df(data)
    out = DataFrame()
    for d in data
        ## Need to transpose matrix again for columnar output
        len = convert(Int32, size(d[1]',1))
        delim = zeros(len)
        Δ = hcat(delim, d[1]')
        header = zeros(DIM + 1)
        header[1] += len
        header[2] += d[2][1]
        header[3] += d[2][2]
        header[4] += d[2][3]
        Δ = vcat(header', Δ)
        Δ = convert(DataFrame, Δ)
        append!(out, Δ)
    end
    return out
end

"""
df_to_array(df::DataFrame; dbId=nothing, required=nothing, reimport=false)
Converts dataframe to array data for training, testing, or evaluation
"""
function df_to_array(df::DataFrame; dbId=nothing, required=nothing, reimport=false)
    dbId ≡ nothing && (dbId = 1)
    dfPolyId = (dbId - 1) * SAMPLE_SIZE + 1
    data = []
    onts = []
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
        !NUMBERS_ONLY && (polyMat = viewdf(df, currentRow+1:currentRow+skip, 2:DIM+1, Array))
        ontology = viewdf(df, currentRow, ontology_index, Int32)
        !reimport && (labels = convert.(Int32,viewdf(df, currentRow:currentRow, 2:4, Array)))
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
                polyMat = vcat([v p v_dual p_dual], polyMat)
            else
                polyMat = viewdf(df, currentRow + 1:currentRow + 4, 1:1, Array)'
            end
        else 
            NUMBERS_ONLY == true && (polyMat = viewdf(df, currentRow + 1:currentRow + 1, 2:5, Array))
        end
        push!(data, (polyMat',ontology))
        !reimport && push!(onts, labels)
    end
    println("Database: ", dbId) 
    println("REQUIRED: $(length(required))")
    println("GOT: $(length(data)) ", length(data) == length(required))
    println("---------------")
    !reimport && (return data, onts)
    return data
end

"""
new_sample(;required = nothing, db_path = DB_PATH,
                     export_path = "None",
                     sample_size = SAMPLE_SIZE,
                     split = false,
                     train_percent = 0.8,
                     db_start = 1,
                     db_end = TOTAL_NUM_DB,
                     get_all=false)
Generates a new data sample from raw database of size sample_size.

Splits data into training and testing sets and exports to a .gzip file at export_path.
"""
function new_sample(;required = nothing, db_path = DB_PATH,
                     export_path = "None",
                     sample_size = SAMPLE_SIZE,
                     split = false,
                     train_percent = 0.8,
                     db_start = 1,
                     db_end = TOTAL_NUM_DB,
                     get_all=false) 
                     
    isnothing(required) && (required = random_sample(sample_size))
    get_all && (required = [x for x in 1:sample_size])

    data = []
    onts = []
    for i in db_start:db_end
        df = loaddf(db_path, i);
        dbData, labels = df_to_array(df, dbId=i, required=required[i])
        append!(data, dbData)
        append!(onts,labels)
    end
    @info("OBTAINED: $(size(data,1)) IN TOTAL")
    
    if export_path != "None"
        if split == true 
            split_export(data, onts, export_path; train_percent=train_percent, csv=required)
        else
            n = 1; while(isdir("$export_path/$n.gz")) n+=1; end
            export_gz(data, export_path, "$n.gz")
        end
    else 
        @warn("DATA NOT EXPORTED: NO export_path GIVEN. RETURNING INSTEAD.")
        return data, onts
    end
end

"""
augment!(data)
Pads data with zeros until each datapoint is of size MAX_SIZE
"""
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

"""
unpackage(data)
Returns data and labels as seperate lists
"""
function unpackage(data)
    X = zeros(DIM*MAX_SIZE, size(data,1))
    Y = []
    for (i,d) in enumerate(data)
        X[:,i] = X[:,i] + d[1][:,]
        append!(Y, d[2])
    end

    return X, Y
end

"""
Conversions between euler number and index.
"""
euler_to_index(x) = convert.(Int32, round.(map(y -> 0.5*(y + 960) + 1, x)))
index_to_euler(x) = convert.(Int32, round.(map(y -> 2*(y - 1) - 960, x)))