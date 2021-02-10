### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ f82d94ae-6a65-11eb-26f8-e5d49d7f7a34
using HTTP, Gumbo, LinearAlgebra

# ╔═╡ 239a1100-6a66-11eb-0382-79b16d8c8dab
function parseCYData(bText)
    polytopeVerts = []
    vertex = []
    skip = false

    for idx in 1:size(bText)[1]

        if skip == true && bText[idx] != ']'
            continue
        end

        if !any(bText[idx] .== [']', 'M', '#'])
            if !any(bText[idx] .== ['\n'])
                push!(vertex, bText[idx])
            end
        elseif bText[idx] == 'M'
            skip = true
            continue
        elseif bText[idx] == '#'
            if size(vertex,1) != 0
                push!(polytopeVerts, vertex)
                vertex = []
            end
        else
            if size(vertex,1) != 0
                for _ in 1:5; pop!(vertex); end;
                push!(polytopeVerts, vertex)
                vertex = []
            end
            skip = false
        end
    end

    popfirst!(polytopeVerts);
    return polytopeVerts;
end;

# ╔═╡ 3b226390-6a66-11eb-3d3e-4b4e7d18360f
mutable struct Model
	layers::AbstractArray
end

# ╔═╡ 86282712-6a6d-11eb-2cdc-9b83e9c5d8db
abstract type AbstractLayer end;

# ╔═╡ 862a980e-6a6d-11eb-223c-639e22f19722
mutable struct linearLayer <: AbstractLayer
	W::AbstractArray
	b::Vector
	depth::Integer
end;

# ╔═╡ 8632ae62-6a6d-11eb-2456-816ff757a015
mutable struct conv2dLayer <: AbstractLayer
	inputShape::Tuple;
	outputShape::Tuple;
	kernelSize::Tuple;
	stride::Integer;
	padding::Integer;
	paddingType::String
	dilation::Integer;
end;

# ╔═╡ 86351f60-6a6d-11eb-2c7e-597eb2cba9b7
mutable struct inceptionLayer <: AbstractLayer
	convA::conv2dLayer
	convB::conv2dLayer
end

# ╔═╡ 2786a580-6b0b-11eb-3797-8751b7d16e19
mutable struct dVar
	
end

# ╔═╡ 5725b6f0-6a66-11eb-2c3b-a3bf242312b2
function Conv2D(nn::Model; kernelSize::Union{Integer, Tuple}, inRank::Tuple, outRank::Tuple, stride::Integer, padding::Integer, paddingType::String, dilation::Integer)
    
end

# ╔═╡ 572827f0-6a66-11eb-2646-7102bc10c403
function Input(nn::Model, layerDepth)
    push!(nn.layers, linearLayer(Diagonal(ones(layerDepth)), zeros(layerDepth), layerDepth))
    println("Shape of Weights_1: $(size(nn.layers[1].W))")
    return x -> x
end

# ╔═╡ 572ce2e0-6a66-11eb-31a8-6dfdd87b9efd
function relu(x)
    x ≥ 0 ? x : 0;
end

# ╔═╡ 9214b3c0-6a6a-11eb-2c3b-6383d9a193ae
function ∘(f,g)
	x -> f(g(x))
end

# ╔═╡ 57354750-6a66-11eb-0bd0-e9d725b778bb
function Compose(funcs...)
	
	F = funcs[2]∘funcs[1]
	
	for id in 3:length(funcs)
		F = funcs[id]∘F
	end
	
	return x -> F(x)
end

# ╔═╡ 71f7d920-6a6d-11eb-196b-c19a4e79ffec
function Compose(f::Function)
	return f∘f
end

# ╔═╡ 71fa4a20-6a6d-11eb-3846-c54de6f5e2fe
function Compose(f::Function, n::Integer)
	
	return
end

# ╔═╡ 697250be-6a66-11eb-2712-93afa98b77fc
# begin
# 	nn = Model(AbstractLayer[])
	
# 	input = Input(nn, 10)
# 	layer1 = Linear(nn, layerDepth = 5, dist="normal", act=relu)
# 	layer2 = Linear(nn, layerDepth = 3, dist="normal", act=relu)
# 	layer3 = Linear(nn, layerDepth = 1, dist="normal", act=relu)
	
# 	network = [input, layer1, layer2, layer3]
	
# 	x = rand(10);
# 	for i in 1:length(network)
# 		println(x)
# 	    x = network[i](x)
# 	end
	
# 	println(x)
# end

# ╔═╡ 99b90710-6a75-11eb-0cad-b5157fba253d
### Automatic Differentiation - Define derivative rules for base operations

begin
	import Base: +, -, *, /, sin, cos, convert, promote_rule	
	
	## Forward-mode
	struct D <: Number
		f::Tuple{Float64, Float64}
	end
	
	+(x::D, y::D) = D((x.f .+ y.f))
	-(x::D, y::D) = D((x.f .- y.f))
	*(x::D, y::D) = D((x.f[1]*y.f[1], x.f[2]*y.f[1] + x.f[1]*y.f[2]))
	/(x::D, y::D) = D((x.f[1] / y.f[1], (x.f[2]*y.f[1] - x.f[1]*y.f[2])/y.f[1]^2))
	convert(::Type{D}, x::Real) = D((x, zero(x)))
	promote_rule(::Type{D}, ::Type{<:Number}) = D
	Base.show(io::IO, x::D) = print(io, x.f[1], "+", x.f[2], "ϵ")
	
	## Reverse-mode
	mutable struct dVar <: Number
		value::Number
		children::AbstractArray
		grad_value::Number
		
		dVar(x) = new(x, Tuple[], NaN)
	end
	
	function +(x::dVar, y::dVar)
		z = dVar(x.value + y.value) 
		push!(x.children, (1, z))
		push!(y.children, (1, z))
		return z
	end
	function -(x::dVar, y::dVar)
		z = dVar(x.value - y.value)
		push!(x.children, (1, z))
		push!(y.children, (-1, z))
		return z
	end
	function *(x::dVar, y::dVar)
		z = dVar(x.value * y.value)
		push!(x.children, (y.value, z))
		push!(y.children, (x.value, z))
		return z
	end
	function /(x::dVar, y::dVar)
		z = dVar(x.value / y.value)
		push!(x.children, (1/y.value, z))
		push!(y.children, (-x.value/y.value^2, z))
		return z
	end
	function sin(x::dVar)
		z = dVar(sin(x.value))
		push!(x.children, (cos(x.value), z))
		return z
	end
	function cos(x::dVar)
		z = dVar(cos(x.value))
		push!(x.children, (-sin(x.value), z))
		return z
	end
	# Try:
	# function /(x::dVar, y::dVar)
	# 	z = x*y^-1
	# end
	
	function Gradient(z::dVar)
		if isnan(z.grad_value)
			z.grad_value = sum(weight * Gradient(var) for (weight, var) in z.children)
		end
		return z.grad_value
	end
	
	*(a::Integer, f::Function) = x -> a*f(x) 
	
end

# ╔═╡ 053cc810-6a66-11eb-2595-056af17241a1
function getCYData(class::String,value::Int64)
    r = HTTP.get("http://quark.itp.tuwien.ac.at/cgi-bin/cy/cydata.cgi?"*class*"="*string(value)*"&L=1000&page=2");
    r_parsed = parsehtml(String(r.body));
    body = r_parsed.root[2];
    return Vector{Char}(body[1][4].text);
end;

# ╔═╡ 2af6d500-6a66-11eb-0f4f-27fb274f52e8
function vectorizeCYData(polytopeVerts, dim)
    polytopes = []
    hull = []

    for p in polytopeVerts
        poly = ""
        for v in p; poly *= v; end
        for e in filter!(x->x!="", split(poly," "))
            push!(hull, parse(Int64, e))
        end
        push!(polytopes, hull)
        hull = []
    end

    out = []
    for p in polytopes
        p = reshape(p, (Int(size(p)[1]/dim), dim))
        push!(out, p')
    end
    
    return out;
end;

# ╔═╡ 48ca7b90-6a66-11eb-181d-e1f4c6268305
function Linear(nn::Model; layerDepth::Integer, dist="uniform", act=relu)
    
    dist == "uniform" ? random = rand : false;
    dist == "normal" ? random = randn : false;
    prevLayerDepth = nn.layers[end].depth;
    
    push!(nn.layers, linearLayer(random(layerDepth,prevLayerDepth), random(layerDepth), layerDepth));
    W = nn.layers[end].W;
    b = nn.layers[end].b;
    print("Shape of Weights_$(length(nn.layers)): $(size(W))")
    return x -> act.(W*x + b);
end

# ╔═╡ 638f6c70-6a6f-11eb-3445-292a5794241d
begin
	nn = Model(AbstractLayer[])
	
	model = Compose( Input(nn, 10),
					 Linear(nn, layerDepth = 5, dist="uniform", act=relu),
					 Linear(nn, layerDepth = 1, dist="uniform", act=relu) )
	
	x = rand(10)
	println(x)
	x = model(x)
	println(x)
end

# ╔═╡ 04972a80-6a6c-11eb-2b86-477790429034
begin
	f = x -> x^2;
	G = x -> 2x;
	h = x -> 2.71^x;
	i = x -> -x;
	
	X = f∘G
	Y = h∘X
	Z = i∘Y
	
	Z(2)
	
	# ((f∘G)∘(h∘i))(x)
end

# ╔═╡ edb99a10-6a6f-11eb-34c8-e36106ef2dba
begin
	τ = 0.1
	e = exp(1)
	f₁ = x -> e^x
	g₁ = x -> -x^2/sqrt(τ)
	
	f₁(0)
end

# ╔═╡ 912781b0-6a77-11eb-351c-817279ff74c3
begin
	x₁ = dVar(0.5)
	x₂ = dVar(4.2)
	z = x₁*x₂ + sin(x₁)
	z.grad_value = 1.0
	
	Gradient(z)
end

# ╔═╡ dbf57080-6a77-11eb-0c79-9d1d271cce23
function Γ(n)
	cnt = n
	xᵢ = 1
	xⱼ = n
	while cnt > 1
		xᵢ = xᵢ*xⱼ 
		xⱼ = xⱼ - 1
		cnt = cnt - 1
	end
	return xᵢ
end

# ╔═╡ Cell order:
# ╠═f82d94ae-6a65-11eb-26f8-e5d49d7f7a34
# ╠═053cc810-6a66-11eb-2595-056af17241a1
# ╠═239a1100-6a66-11eb-0382-79b16d8c8dab
# ╠═2af6d500-6a66-11eb-0f4f-27fb274f52e8
# ╠═3b226390-6a66-11eb-3d3e-4b4e7d18360f
# ╠═86282712-6a6d-11eb-2cdc-9b83e9c5d8db
# ╠═862a980e-6a6d-11eb-223c-639e22f19722
# ╠═8632ae62-6a6d-11eb-2456-816ff757a015
# ╠═86351f60-6a6d-11eb-2c7e-597eb2cba9b7
# ╠═2786a580-6b0b-11eb-3797-8751b7d16e19
# ╠═48ca7b90-6a66-11eb-181d-e1f4c6268305
# ╟─5725b6f0-6a66-11eb-2c3b-a3bf242312b2
# ╟─572827f0-6a66-11eb-2646-7102bc10c403
# ╟─572ce2e0-6a66-11eb-31a8-6dfdd87b9efd
# ╠═9214b3c0-6a6a-11eb-2c3b-6383d9a193ae
# ╟─04972a80-6a6c-11eb-2b86-477790429034
# ╠═57354750-6a66-11eb-0bd0-e9d725b778bb
# ╠═71f7d920-6a6d-11eb-196b-c19a4e79ffec
# ╠═71fa4a20-6a6d-11eb-3846-c54de6f5e2fe
# ╠═697250be-6a66-11eb-2712-93afa98b77fc
# ╠═638f6c70-6a6f-11eb-3445-292a5794241d
# ╠═edb99a10-6a6f-11eb-34c8-e36106ef2dba
# ╠═99b90710-6a75-11eb-0cad-b5157fba253d
# ╠═912781b0-6a77-11eb-351c-817279ff74c3
# ╠═dbf57080-6a77-11eb-0c79-9d1d271cce23
