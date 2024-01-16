using ML_as_MO
using Documenter

DocMeta.setdocmeta!(ML_as_MO, :DocTestSetup, :(using ML_as_MO); recursive=true)

makedocs(;
    modules=[ML_as_MO],
    authors="Joonatan Linkola <joonatan.linkola@gmail.com> and contributors",
    repo="https://github.com/joonatanl/ML_as_MO.jl/blob/{commit}{path}#{line}",
    sitename="ML_as_MO.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://joonatanl.github.io/ML_as_MO.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/joonatanl/ML_as_MO.jl",
    devbranch="main",
)
