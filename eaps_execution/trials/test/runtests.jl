using PyCall
using Pkg
import Base.Filesystem: normpath, joinpath, splitdir
import Base: source_path
import Dates
import StatsBase
import Random


function find_project_root_dir()
    current_dir = abspath(@__DIR__)
    while true
        if isfile(joinpath(current_dir, "Project.toml")) || isfile(joinpath(current_dir, "Manifest.toml"))
            return current_dir
        end
        parent_dir = dirname(current_dir)
        if parent_dir == current_dir
            error("The project root directory could not be found.")
        end
        current_dir = parent_dir
    end
end

homepath = find_project_root_dir()

# Make sure that the correct python library folders are being used for the tests
# (site packages and eta_utility package path).
ENV["PYCALL_JL_RUNTIME_PYTHON"] = joinpath(homepath,".venv","Scripts","python.exe")
pushfirst!(PyVector(pyimport("sys")."path"),joinpath(homepath,".venv", "Lib", "site-packages"),)

include("ProductionScheduling.jl")

test()
