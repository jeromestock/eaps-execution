using JuMP
using JSON
using Plots
using LaTeXStrings


"""
Data used for the parameter estimation for electric sandblasting machine models.
"""
struct SandblastingMachineRegressionData <: RegressionData
    "Name of the electric power consumption timeseries."
    P_el::Symbol
    "Name of the standby mode active indicator (1 or 0) timeseries."
    a_st::Symbol
    "Name of the operational mode active indicator (1 or 0) timeseries."
    a_op::Symbol
    "Name of the working mode active indicator (1 or 0) timeseries."
    a_wk::Symbol
    "Name of the peak mode active indicator (1 or 0) timeseries"
    a_pk::Symbol
    "Regression Periods"
    periods::Union{Nothing, Function}
end


"""
Parameter data for sandblasting machine models, available after parameter estimation.
"""
struct SandblastingMachineParameterData <: ParameterData
    "Regression parameter for electric energy."
    β_el::Vector{Float64}
end


"""
Object describing a sandblasting machine in the factory.
"""
struct SandblastingMachine{T} <: Machine where {T <: ModelData}
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
Instantiate SandblastingMachine object with the names timeseries names for regression.
"""
SandblastingMachine(name; id, P_el, a_st, a_op, a_wk, a_pk, periods) = SandblastingMachine(
    name,
    id,
    1,
    false,
    SandblastingMachineRegressionData(P_el, a_st, a_op, a_wk, a_pk, periods),
)


"""
Instantiate SandblastingMachine object with parameter data from a file.
"""
function SandblastingMachine(id::Int, filename::AbstractString)
    objects = JSON.parsefile(filename)
    for obj in objects
        if obj["resource_id"] == id
            return SandblastingMachine(
                obj["name"],
                obj["resource_id"],
                obj["capacity"],
                obj["unique_job"],
                SandblastingMachineParameterData(
                    obj["parameters"]["beta_el"],
                ),
            )
        end
    end
    return nothing
end


"""
Create a regression model for the machine.

:param machine: The machine object containing the timeseries names (RegressionData).
:param data: Actual timeseries identified by the names in the SandblastingMachine object.
:return: JuMP model.
"""
function regression_model(machine::SandblastingMachine{T}, data::DataFrame) where {T <: RegressionData}

    model = Model()
    names = machine.data

    values = dropmissing(
        (isnothing(names.periods) ? data : subset(data, :Timestamp => names.periods)),
        regression_colnames(names),
    )
    timesteps = size(values)[1]

    # Parameters
    a_st = Vector{Bool}(values[:, names.a_st])
    a_op = Vector{Bool}(values[:, names.a_op])
    a_wk = Vector{Bool}(values[:, names.a_wk])
    a_pk = Vector{Bool}(values[:, names.a_pk])
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
                (a_st[t] + a_op[t] + a_wk[t] + a_pk[t]) * β_el[1] +
                (a_op[t] + a_wk[t] + a_pk[t]) * β_el[2] +
                (a_wk[t] + a_pk[t]) * β_el[3] +
                a_pk[t] * β_el[4] +
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
    machine::SandblastingMachine{T},
    model::JuMP.AbstractModel,
) where {T <: RegressionData}
    vars = object_dictionary(model)

    Dict(
        "type" => "sandblasting",
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
Results vectors from a forward execution of the sandblasting machine model.
"""
struct SandblastingMachineResults <: ModelResults
    "Predicted electric power consumption."
    P_el::Vector{Union{Missing, Float64}}
end


"""
Prediction model for power consumption of the sandblasting machine.

:param machine: The machine object containing the estimated parameters.
:param a_st: Standby mode active indicator (1 or 0) timeseries.
:param a_op: Operational mode active indicator (1 or 0) timeseries.
:param a_wk: Working mode active indicator (1 or 0) timeseries.
:param T_u: External temperature timeseries.
:param T_M1: Machine temperature at the beginning.
"""
function forward_model(
    machine::SandblastingMachine{T};
    a_st,
    a_op,
    a_wk,
    a_pk,
) where {T <: ParameterData}
    β_el = machine.data.β_el

    P_el = (a_st + a_op + a_wk + a_pk) .* β_el[1] + (a_op + a_wk + a_pk) .* β_el[2] + (a_wk + a_pk) .* β_el[3] + a_pk .* β_el[4]

    return SandblastingMachineResults(P_el)
end

"""
Plot for the energy mode. Generic function for plotting the results
and the raw data.

:param values:
:param names:
:param plt_act:
:param x_tickformat:
"""
function plot_state_sandblasting(values, names, x_ticks, x_tickformat; kwargs...)

    plt_act = plot(
        ylabel="Energy state",
        xlabel="Time",
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylims=(0.4, 4.6),
        yticks=([1, 2, 3, 4], ["standby", "operational", "working", "peak"]),
        legend=false,
        xrotation = 45,
    )

    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_st],
        Dict(true => 1),
        orientation=:h,
        bar_width=0.2,
        color=RGB(0.5, 0.5, 0.5),
    )
    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_op],
        Dict(true => 2),
        orientation=:h,
        bar_width=0.2,
        color=RGB(0.5, 0.5, 0.5),
    )
    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_wk],
        Dict(true => 3),
        orientation=:h,
        bar_width=0.2,
        color=RGB(0.5, 0.5, 0.5),
    )
    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_pk],
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
    machine::SandblastingMachine{T},
    data::DataFrame,
    result::SandblastingMachineResults;
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
        "RMSE_standardize" => RMSE/(mean(result.P_el)),
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
        palette=:okabe_ito,
        xrotation = 45,
    )
    plot!(plt_power, values[:, :Timestamp], result.P_el, linewidth=2, color=:black, label="predicted", xrotation = 45)

    plt_act = plot_state_sandblasting(values, names, x_ticks, x_tickformat)

    plt = plot(plt_power, plt_act, layout=@layout([°; °]), xrotation = 45; kwargs...)

    return plt, errors
end


"""
Plot data collected to perform the regression.

:param machine: The machine object containing the timeseries names (RegressionData).
:param data: Actual timeseries identified by the names in the MachineTool object.
:return: A plot object.
"""
function plot_data(machine::SandblastingMachine{T}, data::DataFrame; kwargs...) where {T <: RegressionData}
    l = @layout([°; °])
    names = machine.data

    values =
        coalesce.(
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

    plt_act = plot_state_sandblasting(values, names, x_ticks, x_tickformat)

    plt = plot(plt_power, plt_act, layout=l, xrotation = 45; kwargs...)

    return plt
end
