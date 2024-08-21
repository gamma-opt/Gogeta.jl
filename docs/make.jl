using Documenter
using Gogeta

makedocs(
    modules=[Gogeta],
    authors="Eetu Reijonen, Milana Begantsova",
    sitename="Gogeta.jl",
    format=Documenter.HTML(),
    pages=[
        "Introduction" => "index.md",
        "Features" => [
            "Neural networks" => [
                "General" => "nns_introduction.md",
                "Big-M formulation" => "neural_networks.md",
                "Partition-based formulation" => "psplit_nns.md",
                "Optimization" => "optimization.md",
                "Use as surrogates" => "nns_in_larger.md"
            ],
            "Input convex neural networks" => "icnns.md",
            "Convolutional neural networks" => "cnns.md",
            "Tree ensembles" => "tree_ensembles.md",
        ],
        "Literature" => "literature.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/gamma-opt/Gogeta.jl.git",
    devbranch="main"
)
