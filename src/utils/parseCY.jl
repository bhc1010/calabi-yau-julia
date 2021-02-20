######################################
using DataFrames, CSV, CodecZlib, Mmap

#= Polytope Data from text files: 29×3 DataFrame

│ Row │ Database │ Polytopes │ cumulative_sum │
│     │ Int8     │ Int32     │ Int64          │
├─────┼──────────┼───────────┼────────────────┤
│ 1   │ 5        │ 1560      │ 1560           │
│ 2   │ 6        │ 24188     │ 25748          │
│ 3   │ 7        │ 177445    │ 203193         │
│ 4   │ 8        │ 834637    │ 1037830        │
│ 5   │ 9        │ 2867954   │ 3905784        │
│ 6   │ 10       │ 7725800   │ 11631584       │
│ 7   │ 11       │ 16608386  │ 28239970       │
│ 8   │ 12       │ 29270252  │ 57510222       │
│ 9   │ 13       │ 43457999  │ 100968221      │
│ 10  │ 14       │ 56060583  │ 157028804      │
│ 11  │ 15       │ 64085868  │ 221114672      │
│ 12  │ 16       │ 65615930  │ 286730602      │
│ 13  │ 17       │ 59972681  │ 346703283      │
│ 14  │ 18       │ 48703032  │ 395406315      │
│ 15  │ 19       │ 34847820  │ 430254135      │
│ 16  │ 20       │ 21913679  │ 452167814      │
│ 17  │ 21       │ 12070918  │ 464238732      │
│ 18  │ 22       │ 5826220   │ 470064952      │
│ 19  │ 23       │ 2450719   │ 472515671      │
│ 20  │ 24       │ 898928    │ 473414599      │
│ 21  │ 25       │ 284695    │ 473699294      │
│ 22  │ 26       │ 78467     │ 473777761      │
│ 23  │ 27       │ 18416     │ 473796177      │
│ 24  │ 28       │ 3780      │ 473799957      │
│ 25  │ 29       │ 646       │ 473800603      │
│ 26  │ 30       │ 113       │ 473800716      │
│ 27  │ 31       │ 22        │ 473800738      │
│ 28  │ 32       │ 7         │ 473800745      │
│ 29  │ 33       │ 1         │ 473800746      │  =#

# Construct polytope from the polytope coordinates in verts and append to DataFrame
# of all polytopes (Δ²). Note that the length of the polytope and the classification 
# number are embedded in the DataFrame.
function MakePolytope!(Δ²::DataFrame, verts, ontology::Integer)
    Δ = hcat(verts[1], verts[2], verts[3], verts[4])
    len = convert(Int32,size(Δ, 1))
    delim = zeros(len)
    delim[1] += len
    delim[2] += ontology
    Δ = hcat(delim, Δ)
    Δ = convert(DataFrame, Δ)
    append!(Δ², Δ)
end

# Write Δ² to a .gzip file and compress. Each sub-database 
# has a simple integer naming scheme (sub_db_id). 
function ExportSubDB(Δ²::DataFrame, sub_db_id::Integer)
    write_path = "/media/share/Dev/CalabiYau/data/polytopes_db_4d/$(sub_db_id).gz"
    open(write_path, "w") do io
        stream = GzipCompressorStream(io)
        CSV.write(stream, Δ²)
        close(stream)
        stream = nothing
        GC.gc()
    end #close .gzf
end

# (Corner-case) Catch the first classification number in the file.
function GetFirstOntology(ln::String)
    i = findfirst("[", ln)[1]
    j = findlast("]", ln)[1]
    return parse(Int, ln[i+1:j-1])
end

function DoTheThing()#(a::Integer, b::Integer, polytopeCount::Integer, info::DataFrame)
    Δ² = DataFrame()::DataFrame
    ontology = 0::Integer
    polytopeCount = 0::Integer
    polyVerts = []
    sub_db_id = 1::Integer
    polyPerDB = 0::Integer
    totalPolytopes = 0::Integer
    info = DataFrame(Database = Int8[], Polytopes = Int32[])
    corrupted::Bool = false
    exportCnt = 0
    # help = false

    for id in 5:33
        # Initialize path to current database file
        db = "$id"
        if length(db) == 1  db = "0"*db; end
        data_path = "/media/share/Dev/CalabiYau/data/polytopes_db_4d/src/v$db"
        # Open current database file
        open(data_path) do file 
            line_count = 0
            polyPerDB = 0
            for ln in eachline(file)
                corrupted && !occursin("[", ln) ? continue : corrupted = false; 
                # Are we on the first line?
                if line_count == 0  ontology = GetFirstOntology(ln); line_count += 1; continue; end;
                # Have we finished reading a polytope? Is this a corrupted polytope?
                if line_count % 5 == 0
                    if occursin("[", ln)
                        i = findfirst("[", ln)[1]
                    else
                        corrupted = true;
                        continue;
                    end
                    j = findlast("]", ln)[1]
                    #= Construct polytope from the polytope coordinates in polyVerts, append
                    to DataFrame of all polytopes (Δ²), and reset relevant variables       =#
                    MakePolytope!(Δ², polyVerts, ontology)
                    polyVerts = nothing
                    polytopeCount += 1
                    polyPerDB += 1
                    totalPolytopes += 1
                    polyVerts = []
                    # Are we on the last line?
                    occursin("#", ln) ? break : nothing;
                    #= If Δ² has reached 1,000,000 polytopes, then write Δ² to file. After 
                    compression, iterate the sub-database identifer and parse the next 
                    classification number (ontology) from the current line.             =#
                    if polytopeCount == 1000000
                        ExportSubDB(Δ², sub_db_id)
                        exportCnt += 1
                        Δ² = nothing
                        sub_db_id += 1
                        polytopeCount = 0
                        # if exportCnt == 5
                        #     help = true
                        #     return nothing
                        # end
                        Δ² = DataFrame()
                    end #end polytope count check
                    
                    ontology = parse(Int, ln[i+1:j-1])
                else # if reading a coordinate line
                    row = [x for x in parse.(Int16, split(strip(ln)))]
                    push!(polyVerts, row)
                    row = nothing
                end #end current line
                line_count += 1 
            end #end eachline
            current_db_info = DataFrame(Database = parse(Int8,db), Polytopes = polyPerDB)
            append!(info, current_db_info)
            run(`clear`)
            println(info)
        end #close file
    end

    ExportSubDB(Δ², sub_db_id)

    info.cumulative_sum = cumsum(info.Polytopes)
    run(`clear`)
    println(info)
    # return polytopeCount, info
end

DoTheThing()



### Need to remove corrupted polytope at line 7289811 - 7289832 

# (tmppath, tmpio) = mktemp()
# data_path = "/media/share/Dev/CalabiYau/data/polytopes_db_4d/v21"
# # Open current database file
# open(data_path) do file
#     cnt = 0
#     for ln in eachline(file)
#         cnt += 1
#         if 7289811 ≤ cnt ≤ 7289832
#             continue
#         end
#         println(tmpio, ln)
#     end
# end
# close(tmpio)
# mv(tmppath, "/media/share/Dev/CalabiYau/data/polytopes_db_4d/v21", force=true)    
