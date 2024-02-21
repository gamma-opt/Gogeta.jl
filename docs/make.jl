using Documenter
using Gogeta

makedocs(
    modules=[Gogeta],
    authors="Eetu Reijonen",
    sitename="Gogeta.jl",
    format= Documenter.HTML(),
    pages=[
        "Introduction" => "index.md",
        "Public API" => "api.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/gamma-opt/Gogeta.jl.git",
    devbranch="main"
)
