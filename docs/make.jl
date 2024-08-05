using Documenter
using Gogeta

makedocs(
    modules=[Gogeta],
    authors="Eetu Reijonen",
    sitename="Gogeta.jl",
    format=Documenter.HTML(),
    pages=[
        "Introduction" => "index.md",
        "Tree ensembles" => "tree_ensembles.md",
        "Neural networks" => "neural_networks.md",
        "Neural networks in larger optimization problems" => "nns_in_larger.md",
        "Input convex neural networks" => "icnns.md",
        "Public API" => "api.md",
        "Literature" => "literature.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/gamma-opt/Gogeta.jl.git",
    devbranch="main"
)
