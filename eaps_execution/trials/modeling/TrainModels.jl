import Plots: gr, plot, savefig, RGBA
import Impute: interp, locf

using CPLEX
using DataFrames
using JSON
using JuMP
using Logging
using Statistics
using Dates
using Base.Filesystem

using ProductionScheduling

function main()
    homepath = @__DIR__
    model_export_path = joinpath(homepath, "scenario", "productionsystem", "machines.json")
    errors_export_path = joinpath(homepath, "scenario", "productionsystem", "machines_models_errors_measurement.json")
    data_path = joinpath(dirname(dirname(@__DIR__)), "preprocessing", "src", "data")
    plot_path = "results/dataplots/"
    plot_suffix = "pdf"

    plots_kwargs = (
        :plot_titlefontsize => 16,
        :annotation_fontsize => 16,
        :guidefont => 16,
        :ytickfont => 14,
        :xtickfont => 14,
        :legendfont => 14,
        :fontfamily => "Helvetica",
        :framestyle => :box,
        :grid => true,
        :formatter => :plain,
        :legend_background_color =>RGBA(1, 1, 1, 0.7),
        :margin => (1, :mm),
        :top_margin => (1, :mm),
        :bottom_margin => (2, :mm),
        :right_margin => (7, :mm),
        :left_margin => (8, :mm),
        :dpi =>150,
        :size => (
            1.75 * 321.51616 * 120 * 0.01384, # width in pt * dpi * inches per pt
            1.175 * 469.4704 * 120 * 0.01384 * 0.95, # heigth in pt * dpi * inches per pt * 0.95
        ),
    )

    # Make sure that the destination folder exists or create it if it does not yet exist
    mkpath(joinpath(homepath, plot_path))

    # Make new model_export_path file
    if isfile(model_export_path)
        rm(model_export_path)
        println("-----MODEL EXPORT FILE WAS DELETED AND WILL BE RECREATED-----")
    else
        println("-----MODEL EXPORT FILE DOESN'T EXISTS AND WILL BE CREATED-----")
    end

    # Make new errors_export_path file
    if isfile(errors_export_path)
        rm(errors_export_path)
        println("-----ERRORS EXPORT FILE WAS DELETED AND WILL BE RECREATED-----")
    else
        println("-----ERRORS EXPORT FILE DOESN'T EXISTS AND WILL BE CREATED-----")
    end

    # Use gr plotting backend
    gr()

    machines = ["gmb", "gmd", "pbc", "gic"]

    for machine in machines

        println("-----START PLOTTING FOR MACHINE $machine-----")

        plot_modeldata(
            machine,
            NamedTuple(),
            plot_path,
            "plot_0129",
            plots_kwargs,
            plot_suffix,
            data_path,
            "$(machine)/$(machine)_2024-01-29_rTotalActivePower.csv"
        )

        plot_modeldata(
            machine,
            NamedTuple(),
            plot_path,
            "plot_0130",
            plots_kwargs,
            plot_suffix,
            data_path,
            "$(machine)/$(machine)_2024-01-30_rTotalActivePower.csv"
        )

        plot_modeldata(
            machine,
            NamedTuple(),
            plot_path,
            "plot_0131",
            plots_kwargs,
            plot_suffix,
            data_path,
            "$(machine)/$(machine)_2024-01-31_rTotalActivePower.csv"
        )

        println("-----START TRAINING FOR MACHINE $machine-----")

        train_models(
            machine,
            NamedTuple(),
            plot_path,
            plots_kwargs,
            plot_suffix,
            model_export_path,
            data_path,
            "$(machine)/$(machine)_2024-01-29_rTotalActivePower.csv",
            "$(machine)/$(machine)_2024-01-31_rTotalActivePower.csv",
            "$(machine)/$(machine)_2024-02-01_rTotalActivePower.csv",
            )

        println("-----START TESTING FOR MACHINE $machine-----")

        test_models(
            machine,
            NamedTuple(),
            plot_path,
            plots_kwargs,
            plot_suffix,
            model_export_path,
            errors_export_path,
            data_path,
            "$(machine)/$(machine)_2024-01-30_rTotalActivePower.csv",
        )
    end
end

"""
Import data and create plots with it.
"""
function plot_modeldata(
    machine,
    periods,
    plot_path,
    plot_prefix,
    plots_kwargs,
    plot_suffix,
    data_root_path,
    data_files...,
)
    # Import Data from experiments
    data = import_experimentdata(data_root_path, data_files...)
    # describe_data(data)
    models, data = create_models!(data, periods, machine)

    # Plot measured data
    map(
        m -> savefig(
        plot_data(m, data; plots_kwargs...),
        joinpath(@__DIR__, plot_path, "$(plot_prefix)_$(m.id)_$(m.name).$plot_suffix"),
),
        values(models),
    )
end

    """
    Estimate machine energy model parameters with provided data.
    """
    function train_models(
        machine,
        periods,
        plot_path,
        plots_kwargs,
        plot_suffix,
        model_export_file,
        data_root_path,
        data_files...,
    )

    # Import Data from experiments
    data = import_experimentdata(data_root_path, data_files...)
    # describe_data(data)

    model, data = create_models!(data, periods, machine)

    # Plot measured data
    map(
        m -> savefig(
        plot_data(m, data; plots_kwargs...),
        joinpath(@__DIR__, plot_path, "training_$(m.id)_$(m.name).$plot_suffix"),
        ),
        values(model),
    )

    # Create and optimize regression model
    regression_models = Dict(map(m -> m.id => regression_model(m, data), values(model)))

    optimizer = prepare_optimization()
    optimizemodels!(regression_models, optimizer; display=true) # outfile="model.txt",

    if isfile(model_export_file)
        # File already exists, read the content
        json = JSON.parsefile(model_export_file; use_mmap=false)
    else
        # File does not exist, create an empty array
        json = []
    end

    # Add the new parameters
    append!(json, collect(map(m -> export_parameters(m, regression_models[m.id]), values(model))))

    # Export model data as resource file for scheduling
    open(normpath(model_export_file), "w") do f
        JSON.print(f, json, 4)
    end
end

"""
Test model fit compared to provided test data.
"""
function test_models(
    machine,
    periods,
    plot_path,
    plots_kwargs,
    plot_suffix,
    model_import_file,
    errors_export_path,
    data_root_path,
    data_files...,
    )

    error = Dict()

    if isfile(errors_export_path)
        # File already exists, read the content
        json = JSON.parsefile(errors_export_path; use_mmap=false)
    else
        # File does not exist, create an empty array
        json = []
    end

    # Import Data from experiments
    data = import_experimentdata(data_root_path, data_files...)

    models, data = create_models!(data, periods, machine)
    parametrized_models = map(m -> convert_model(m, model_import_file), values(models))

    results = forward_models(parametrized_models, models, data)

    # Plot measured data and print errors measures.
    for m in values(models)
        plt, errors = plot_result(m, data, results[m.id]; plots_kwargs...)

        savefig(plt, joinpath(@__DIR__, plot_path, "test_$(m.id)_$(m.name).$plot_suffix"))

        append!(json, Dict(m.name => errors))

        # Export errors data as file
        open(normpath(errors_export_path), "w") do f
            JSON.print(f, json, 4)
        end

        for (name, value) in pairs(errors)
            error["$(m.id)_$(m.name) $name"] = value
            println("$(m.id)_$(m.name) $name: $value")
        end
    end
end

"""
Create models that can be parametrized
"""
function create_models!(data, periods, machine)
    # Create Machine model
    if machine == "pbc"
        pbc, data = model_pbc!(data, name="PBC", id=971, periods=periods)
        model = Dict(pbc.id => pbc)
    elseif machine == "gmd"
        gmd, data = model_laser_cutting!(data, :gmd; name="GMD", id=972, periods=periods)
        model = Dict(gmd.id => gmd)
    elseif machine == "gmb"
        gmb, data = model_laser_cutting!(data, :gmb; name="GMB", id=973, periods=periods)
        model = Dict(gmb.id => gmb)
    elseif machine == "gic"
        gic, data = model_gic!(data, name="GIC", id=974, capacity=2, periods=periods)
        model = Dict(gic.id => gic)
    end

    return model, data
end

"""
Create models from stored parameters (for testing).
"""
convert_model(model::LaserCuttingMachine, filename) = LaserCuttingMachine(model.id, filename)

"""
Create models from stored parameters (for testing).
"""
convert_model(model::SandblastingMachine, filename) = SandblastingMachine(model.id, filename)

"""
Create models from stored parameters (for testing).
"""
convert_model(model::FlameCuttingMachine, filename) = FlameCuttingMachine(model.id, filename)

"""
Execute predictions with parametrized models.
"""
function forward_models(parametrized_models::Vector, regression_models::Dict, data::DataFrame)
    results = Dict{Int, ModelResults}()
    for model in parametrized_models
        names = regression_models[model.id].data
        values = dropmissing(
            isnothing(names.periods) ? data : subset(data, :Timestamp => names.periods),
            union(regression_colnames(names), (:Timestamp,)),
        )

        if typeof(model) <: LaserCuttingMachine
            results[model.id] = forward_model(
                model,
                a_st2=values[:, names.a_st2],
                a_st1=values[:, names.a_st1],
                a_op=values[:, names.a_op],
                a_wk=values[:, names.a_wk],
            )
        elseif typeof(model) <: SandblastingMachine
            results[model.id] = forward_model(
                model,
                a_st=values[:, names.a_st],
                a_op=values[:, names.a_op],
                a_wk=values[:, names.a_wk],
                a_pk=values[:, names.a_pk]
            )
        elseif typeof(model) <: FlameCuttingMachine
            results[model.id] = forward_model(
                model,
                a_st=values[:, names.a_st],
                a_op=values[:, names.a_op],
                a_wk=values[:, names.a_wk]
            )
        else
            error("Unknown model type: $typeof(model).")
        end
    end
    return results
end

"""
Preprocess data and create a model for parameter estimation of a generic lasercutting machine.
"""
function model_laser_cutting!(data, machine_type; name, id, periods)
    # Model setup for the specified laser cutting machine type
    periods_key = Symbol(machine_type)
    periods = haskey(periods, periods_key) ? periods[periods_key] : nothing

    return LaserCuttingMachine(
        name,
        id=id,
        P_el=:watt,
        a_st2=:st2,
        a_st1=:st1,
        a_op=:op,
        a_wk=:wk,
        periods=periods,
    ),
    data
end

"""
Preprocess data and create a model for parameter estimation for the sandblasting machine pbc.
"""
function model_pbc!(data; name, id, periods)
    # Model setup for the PBC sandblasting machine
    periods = haskey(periods, :pbc) ? periods[:pbc] : nothing

    return SandblastingMachine(
        name,
        id=id,
        P_el=:watt,
        a_st=:st,
        a_op=:op,
        a_wk=:wk,
        a_pk=:pk,
        periods=periods,
    ),
    data
end

"""
Preprocess data and create a model for parameter estimation of a generic lasercutting machine.
"""
function model_gic!(data; name, id, capacity, periods)
    # Model setup for the gic flame cutting machine
    periods = haskey(periods, :gic) ? periods[:gic] : nothing

    return FlameCuttingMachine(
        name,
        id=id,
        capacity=capacity,
        P_el=:watt,
        a_st=:st,
        a_op=:op,
        a_wk=:wk,
        periods=periods,
    ),
    data
end

"""
Prepare the model parameter estimation optimization.
"""
function prepare_optimization()
    global_logger(ConsoleLogger(stdout, Logging.Info; show_limited=true))
    # CPLEX Optimizer
    optimizer_with_attributes(CPLEX.Optimizer, "CPX_PARAM_SCRIND" => 1, "CPX_PARAM_BARDISPLAY" => 2)
end

main()
