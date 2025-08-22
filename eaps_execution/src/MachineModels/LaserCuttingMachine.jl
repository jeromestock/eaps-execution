using JuMP
using JSON
using Plots
using LaTeXStrings


"""
Data used for the parameter estimation for electric laser cutting machine models.
"""
struct LaserCuttingMachineRegressionData <: RegressionData
    "Name of the electric power consumption timeseries."
    P_el::Symbol
    "Name of the standby1 mode active indicator (1 or 0) timeseries."
    a_st2::Symbol
    "Name of the standby1 mode active indicator (1 or 0) timeseries."
    a_st1::Symbol
    "Name of the operational mode active indicator (1 or 0) timeseries."
    a_op::Symbol
    "Name of the working mode active indicator (1 or 0) timeseries."
    a_wk::Symbol
    "Regression Periods"
    periods::Union{Nothing, Function}
end

"""
Parameter data for laser cutting machine models, available after parameter estimation.
"""
struct LaserCuttingMachineParameterData <: ParameterData
    "Regression parameter for electric energy."
    β_el::Vector{Float64}
end

"""
Object describing a laser cutting machine in the factory.
"""
struct LaserCuttingMachine{T} <: Machine where {T <: ModelData}
    "Name of the machine."
    name::String
    "Unique identifier of the machine."
    id::Int
    "Capacity of the machine (how many parts it can produce at once)."
    capacity::Int
    "Unique job flag (true if all coinciding jobs have to be the same operation.)"
    unique_job::Bool
    "Either parameter or regression data object."
    data::T
end

"""
Instantiate laser cutting machine object with the names timeseries names for regression.
"""
LaserCuttingMachine(name; id, P_el, a_st2, a_st1, a_op, a_wk, periods) = LaserCuttingMachine(
    name,
    id,
    1,
    true,
    LaserCuttingMachineRegressionData(P_el, a_st2, a_st1, a_op, a_wk, periods),
)

"""
Instantiate laser cutting machine object with parameter data from a file.
"""
function LaserCuttingMachine(id::Int, filename::AbstractString)
    objects = JSON.parsefile(filename)
    for obj in objects
        if obj["resource_id"] == id
            return LaserCuttingMachine(
                obj["name"],
                obj["resource_id"],
                obj["capacity"],
                obj["unique_job"],
                LaserCuttingMachineParameterData(
                    obj["parameters"]["beta_el"],
                ),
            )
        end
    end
    return nothing
end

"""
Create a regression model for the laser cutting machine.

:param machine: The machine object containing the timeseries names (RegressionData).
:param data: Actual timeseries identified by the names in the LaserCuttingMachine object.
:return: JuMP model.
"""
function regression_model(machine::LaserCuttingMachine{T}, data::DataFrame) where {T <: RegressionData}

    model = Model()
    names = machine.data

    values = dropmissing(
        (isnothing(names.periods) ? data : subset(data, :Timestamp => names.periods)),
        regression_colnames(names),
    )
    timesteps = size(values)[1]

    # Parameters
    a_st2 = Vector{Bool}(values[:, names.a_st2])
    a_st1 = Vector{Bool}(values[:, names.a_st1])
    a_op = Vector{Bool}(values[:, names.a_op])
    a_wk = Vector{Bool}(values[:, names.a_wk])
    P_el = Vector{Float64}(values[:, names.P_el])

    # Variables for electric power consumption
    @variables(model, begin
        β_el[1:4]
        ϵ_el[1:timesteps]
    end)

    @constraints(
        model,
        begin
            c_el[t=1:timesteps],
            P_el[t] == (
                (a_st2[t] + a_st1[t] + a_op[t] + a_wk[t]) * β_el[1] +
                (a_st1[t] + a_op[t] + a_wk[t]) * β_el[2] +
                (a_op[t] + a_wk[t]) * β_el[3] +
                a_wk[t] * β_el[4] +
                ϵ_el[t]
            )
        end
    )

    @constraints(model, begin
        β_el[1:4] .>= 0
    end)

    # Objective for the regression model is to reduce the quadratic loss function.
    @objective(model, Min, sum(ϵ_el[t]^2 for t in 1:timesteps))
    return model
end

"""
Export the parameters identified during the regression.

:param machine: The machine object containing the timeseries names (RegressionData).
:param model: The solved JuMP model containing the estimated parameters.
:return: Dictionary to be exported to a JSON file.
"""
function export_parameters(
    machine::LaserCuttingMachine{T},
    model::JuMP.AbstractModel,
) where {T <: RegressionData}
    vars = object_dictionary(model)

    Dict(
        "type" => "lasercutting",
        "name" => machine.name,
        "resource_id" => machine.id,
        "unique_job" => machine.unique_job,
        "capacity" => machine.capacity,
        "parameters" => Dict(
            "beta_el" => value.(vars[:β_el]),
        ),
    )
end

"""
Results vectors from a forward execution of the laser cutting machine model.
"""
struct LaserCuttingMachineResults <: ModelResults
    "Predicted electric power consumption."
    P_el::Vector{Union{Missing, Float64}}
end

"""
Prediction model for power consumption of the laser cutting machine model.

:param machine: The machine object containing the estimated parameters.
:param a_st2: Standby mode active indicator (1 or 0) timeseries.
:param a_st1: Standby* mode active indicator (1 or 0) timeseries.
:param a_op: Operational mode active indicator (1 or 0) timeseries.
:param a_wk: Working mode active indicator (1 or 0) timeseries.
"""
function forward_model(
    machine::LaserCuttingMachine{T};
    a_st2,
    a_st1,
    a_op,
    a_wk,
) where {T <: ParameterData}
    β_el = machine.data.β_el

    P_el = (a_st2 + a_st1 + a_op + a_wk) .* β_el[1] +
     (a_st1 + a_op + a_wk) .* β_el[2] +
     (a_op + a_wk) .* β_el[3] +
     a_wk .* β_el[4]

    return LaserCuttingMachineResults(P_el)
end

"""
Plot for the energy mode. Generic function for plotting the results
and the raw data.

:param values:
:param names:
:param plt_act:
:param x_tickformat:
"""
function plot_state_laser(values, names, x_ticks, x_tickformat; kwargs...)

    plt_act = plot(
        ylabel="Energy state",
        xlabel="Time",
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylims=(0.4, 4.6),
        yticks=([1, 2, 3, 4], ["standby", "standby*", "operational", "working"]),
        legend=false,
        xrotation = 45,
    )

    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_st2],
        Dict(true => 1),
        orientation=:h,
        bar_width=0.2,
        color=RGB(0.5, 0.5, 0.5),
    )
    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_st1],
        Dict(true => 2),
        orientation=:h,
        bar_width=0.2,
        color=RGB(0.5, 0.5, 0.5),
    )
    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_op],
        Dict(true => 3),
        orientation=:h,
        bar_width=0.2,
        color=RGB(0.5, 0.5, 0.5),
    )
    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_wk],
        Dict(true => 4),
        orientation=:h,
        bar_width=0.2,
        color=RGB(0.5, 0.5, 0.5),
    )

    return plt_act
end

"""
Results plot for a prediction made using the machine energy model.

:param machine: The machine object containing the timeseries names (RegressionData).
:param data: Actual timeseries identified by the names in the MachineTool object.
:param results: Results vectors from a forward execution of the machine tool model.
"""
function plot_result(
    machine::LaserCuttingMachine{T},
    data::DataFrame,
    result::LaserCuttingMachineResults;
    kwargs...,
) where {T <: RegressionData}
    names = machine.data
    values = dropmissing(
        isnothing(names.periods) ? data : subset(data, :Timestamp => names.periods),
        union(regression_colnames(names), (:Timestamp,)),
    )

    # Calculate error measures
    RMSE = sqrt(sum((result.P_el .- values[:, names.P_el]) .^ 2) / length(result.P_el))
    MAE = sum(abs.(result.P_el .- values[:, names.P_el])) / length(result.P_el)

    errors = Dict{String, Float64}(
        "RMSE" => RMSE,
        "NRMSE" => RMSE/(mean(result.P_el)),
        "MAE" => MAE,
        "TotalEnergyPercentageError_P_el" => (sum(result.P_el) - sum(values[:, names.P_el])) / sum(values[:, names.P_el]),
    )

    x_ticks = range(ceil(values[1, :Timestamp], Dates.Hour), floor(values[end, :Timestamp], Dates.Hour), step=Hour(2))
    x_tickformat = Dates.format.(x_ticks, "dd. u HH:MM")

    # format y-axis with LaTeXStrings
    # y_formatter(value, _index) = "\$$(round(value, digits=0))\$"

    plt_power = plot(
        values[:, :Timestamp],
        values[:, names.P_el],
        color=RGB(0.5, 0.5, 0.5),
        ylabel="Electric power in W",
        xlabel="Time",
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        label="actual",
        # yformatter=y_formatter,
        palette=:okabe_ito,
    )
    plot!(plt_power, values[:, :Timestamp], result.P_el, linewidth=2, color=:black, label="predicted", xrotation = 45)

    plt_act = plot_state_laser(values, names, x_ticks, x_tickformat)

    plt = plot(plt_power, plt_act, layout=@layout([°; °]), xrotation = 45; kwargs...)

    return plt, errors
end

"""
Plot data collected to perform the regression.

:param machine: The machine object containing the timeseries names (RegressionData).
:param data: Actual timeseries identified by the names in the MachineTool object.
:return: A plot object.
"""
function plot_data(machine::LaserCuttingMachine{T}, data::DataFrame; kwargs...) where {T <: RegressionData}
    l = @layout([°; °])
    names = machine.data

    values = coalesce.(
        (isnothing(names.periods) ? data : subset(data, :Timestamp => names.periods))[
            !,
            union(regression_colnames(names), (:Timestamp,)),
        ],
        NaN64,
    )

    time_range = values[end, :Timestamp] - values[1, :Timestamp]
    duration_days = Dates.value(time_range)/ (24 * 60 * 60 * 1000)
    duration_days = round(Int, duration_days)

    x_ticks = range(ceil(values[1, :Timestamp], Dates.Hour), floor(values[end, :Timestamp], Dates.Hour), step=Hour(2*duration_days))
    x_tickformat = Dates.format.(x_ticks, "dd. u HH:MM")

    # format y-axis with LaTeXStrings
    # y_formatter(value, _index) = "\$$(round(value, digits=0))\$"

    plt_power = plot(
        values[:, :Timestamp],
        values[:, names.P_el],
        color=RGB(0.5, 0.5, 0.5),
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylabel="Electric power in W",
        xlabel="Time",
        label="electric power",
        # yformatter=y_formatter,
        palette=:okabe_ito,
        legend=false,
        xrotation = 45,
    )

    plt_act = plot_state_laser(values, names, x_ticks, x_tickformat)

    plt = plot(plt_power, plt_act, layout=l, xrotation = 45; kwargs...)

    return plt
end
