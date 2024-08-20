using Documenter
using Gogeta

makedocs(
    modules=[Gogeta],
    authors="Eetu Reijonen",
    sitename="Gogeta.jl",
    format=Documenter.HTML(),
    pages=[
        "Introduction" => "index.md",
        "Tutorials" => [
            "Neural networks" => [
                "Practicalities related to NNs" => "nns_introduction.md",
                "Big-M formulation of NNs" => "neural_networks.md",
                "Psplit formulation of NNs" => "psplit_nns.md",
                "Optimization of formulation" => "optimization.md",
                "Neural networks in larger optimization problems" => "nns_in_larger.md",
                "Input convex neural networks" => "icnns.md",
            ],
            "CNNS" => "cnns.md",
            "Tree ensembles" => "tree_ensembles.md",
        ],
        "Public API" => "api.md",
        "Literature" => "literature.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/gamma-opt/Gogeta.jl.git",
    devbranch="main"
)
