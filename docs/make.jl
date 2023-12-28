using Gogeta
using Documenter

DocMeta.setdocmeta!(Gogeta, :DocTestSetup, :(using Gogeta); recursive=true)

makedocs(;
    modules=[Gogeta],
    authors="Nikita Belyak <nikita.belyak1994@gmail.com> and contributors",
    sitename="Gogeta.jl",
    pages=[
        "Home" => "index.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/gamma-opt/Gogeta.jl.git",
    devbranch="main",
)
