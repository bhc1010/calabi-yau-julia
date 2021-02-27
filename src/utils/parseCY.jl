######################################
using DataFrames, CSV, CodecZlib, Mmap

#= Polytope Data from text files: 29×3 DataFrame
│ Row │ Database │ Polytopes │ cumulative_sum │
│     │ Int8     │ Int32     │ Int64          │
├─────┼──────────┼───────────┼────────────────┤
│ 1   │ 5        │ 1561      │ 1561           │
│ 2   │ 6        │ 24189     │ 25750          │
│ 3   │ 7        │ 177446    │ 203196         │
│ 4   │ 8        │ 834638    │ 1037834        │
│ 5   │ 9        │ 2867955   │ 3905789        │
│ 6   │ 10       │ 7725801   │ 11631590       │
│ 7   │ 11       │ 16608387  │ 28239977       │
│ 8   │ 12       │ 29270253  │ 57510230       │
│ 9   │ 13       │ 43458000  │ 100968230      │
│ 10  │ 14       │ 56060584  │ 157028814      │
│ 11  │ 15       │ 64085869  │ 221114683      │
│ 12  │ 16       │ 65615931  │ 286730614      │
│ 13  │ 17       │ 59972682  │ 346703296      │
│ 14  │ 18       │ 48703033  │ 395406329      │
│ 15  │ 19       │ 34847821  │ 430254150      │
│ 16  │ 20       │ 21913680  │ 452167830      │
│ 17  │ 21       │ 12070919  │ 464238749      │
│ 18  │ 22       │ 5826221   │ 470064970      │
│ 19  │ 23       │ 2450720   │ 472515690      │
│ 20  │ 24       │ 898929    │ 473414619      │
│ 21  │ 25       │ 284696    │ 473699315      │
│ 22  │ 26       │ 78468     │ 473777783      │
│ 23  │ 27       │ 18417     │ 473796200      │
│ 24  │ 28       │ 3781      │ 473799981      │
│ 25  │ 29       │ 647       │ 473800628      │
│ 26  │ 30       │ 114       │ 473800742      │
│ 27  │ 31       │ 23        │ 473800765      │
│ 28  │ 32       │ 8         │ 473800773      │
│ 29  │ 33       │ 2         │ 473800775      │   =#

const DIM = 4

# Construct polytope from the polytope coordinates in verts and append to DataFrame
# of all polytopes (Δ²). Note that the length of the polytope and the classification 
# number are embedded in the DataFrame.
function MakePolytope!(Δ²::DataFrame, verts, ontology)
    Δ = hcat(verts...)
    size(Δ,1) < size(Δ,2) ? Δ = Δ' : nothing;
    len = convert(Int32,size(Δ, 1))
    delim = zeros(len)
    delim[1] += ontology.v
    delim[2] += ontology.p
    delim[3] += ontology.v_dual
    delim[4] += ontology.p_dual
    Δ = hcat(delim, Δ)
    header = zeros(1 + 4)
    header[1] = len
    header[2] = ontology.h11
    header[3] = ontology.h21
    header[4] = ontology.euler
    Δ = vcat(header', Δ)
    Δ = convert(DataFrame, Δ)
    println(Δ)
    run(`quit`)
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

# Grab feature data
function GetOntology(ln::String)
    info =  split(strip(ln))
    Δ⁰_pnts = parse(Int,split(info[3], ":")[2])
    Δ⁰_vrts = parse(Int,info[4])
    Δ_pnts = parse(Int,split(info[5], ":")[2])
    Δ_vrts = parse(Int,info[6])
    H = split(split(info[7], ":")[2], ",")
    h11 = parse(Int,H[1])
    h21 = parse(Int,H[2])
    euler = parse(Int,split(split(info[end], "[")[2],"]")[1])
    return (p_dual = Δ⁰_pnts, v_dual = Δ⁰_vrts,p = Δ_pnts, v =  Δ_vrts,h11= h11,h21= h21,euler= euler)
end

function DoTheThing()#(a::Integer, b::Integer, polytopeCount::Integer, info::DataFrame)
    Δ² = DataFrame()::DataFrame
    ontology = ()
    polytopeCount = 0::Integer
    verts = []
    sub_db_id = 1::Integer
    polyPerDB = 0::Integer
    info = DataFrame(Database = Int8[], Polytopes = Int32[])

    for id in 5:33
        # Initialize path to current database file
        db = "$id"
        if length(db) == 1  db = "0"*db; end
        data_path = "/media/share/Dev/CalabiYau/data/polytopes_db_4d/src/v$db"
        # Open current database file
        open(data_path) do file 
            firstLine = true
            polyPerDB = 0
            for ln in eachline(file)
                # Are we on the first line?
                if firstLine
                    ontology = GetOntology(ln)
                    firstLine = false
                    continue
                end
                # Are we on the last line?
                if occursin("#", ln)
                    MakePolytope!(Δ², verts, ontology)
                    verts = nothing
                    polytopeCount += 1
                    polyPerDB += 1
                    verts = []
                    break
                end
                # Have we finished reading a polytope?
                if occursin("[", ln)
                    #= Construct polytope from the polytope coordinates in verts, append
                    to DataFrame of all polytopes (Δ²), and reset relevant variables       =#
                    MakePolytope!(Δ², verts, ontology)
                    verts = nothing
                    polytopeCount += 1
                    polyPerDB += 1
                    verts = []
                    #= If Δ² has reached 1,000,000 polytopes, then write Δ² to file. After 
                    compression, iterate the sub-database identifer and parse the next 
                    classification number (ontology) from the current line.             =#
                    if polytopeCount == 1000000
                        ExportSubDB(Δ², sub_db_id)
                        Δ² = nothing
                        sub_db_id += 1
                        polytopeCount = 0
                        Δ² = DataFrame()
                    end #end polytope count check
                    ontology = GetOntology(ln)
                else # if reading a coordinate line
                    row = [x for x in parse.(Int16, split(strip(ln)))]
                    push!(verts, row)
                end #end current line
            end #end eachline
            current_db_info = DataFrame(Database = id, Polytopes = polyPerDB)
            append!(info, current_db_info)
            run(`clear`)
            println(info)
        end #close file
    end
    ## Export final (partial) database
    ExportSubDB(Δ², sub_db_id)

    info.cumulative_sum = cumsum(info.Polytopes)
    run(`clear`)
    println(info)
end

DoTheThing()   
