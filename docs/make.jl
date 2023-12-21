using Gogeta
using Documenter

DocMeta.setdocmeta!(Gogeta, :DocTestSetup, :(using Gogeta); recursive=true)

makedocs(;
    modules=[Gogeta],
    authors="Nikita Belyak <nikita.belyak1994@gmail.com> and contributors",
    repo="https://github.com/'Nikita Belyak'/Gogeta.jl/blob/{commit}{path}#{line}",
    sitename="Gogeta.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://'Nikita Belyak'.github.io/Gogeta.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/'Nikita Belyak'/Gogeta.jl",
    devbranch="main",
)
