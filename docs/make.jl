using Gogeta
using Documenter

makedocs(
    modules=[Gogeta],
    authors="Nikita Belyak <nikita.belyak1994@gmail.com> and contributors",
    sitename="Gogeta.jl",
    format= Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/gamma-opt/Gogeta.jl.git",
    devbranch="main",
)
