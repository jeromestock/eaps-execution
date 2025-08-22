using ProductionScheduling
import ProductionScheduling: generate_expert_schedules
using DataFrames
using Plots
using PyCall
using Dates
using JSON
using CSV
import ColorSchemes
import Base.Threads: @threads, nthreads
import Graphs: outdegree, outneighbors
import Impute: interp

@pyimport numpy
@pyimport eta_utility as etautility

gr()

"""
Environment class representing the Factory environment
"""
mutable struct Environment <: ProductionEnvironment
    "Settings for the optimization run."
    settings::EnvironmentSettings
    "Status of the optimization run."
    status::EnvironmentStatus

    "Structure of the production system."
    structure::ProductionSystem
    "Events for the production scheduling problem."
    events::EventsMaps

    "Buffer of the rewards for the last set of solutions."
    buffer_rewards::Matrix{Float64}
    "Buffer of the events for the last set of solutions."
    buffer_events::Matrix{Int}
    "Buffer of the variables for the last set of solutions."
    buffer_variables::Matrix{Int}
    "Buffer of the solutions sorted by front."
    buffer_fronts::Union{Vector{Vector{Int}}, Matrix{Int}}

    "DataFrame containing some scenario data like energy prices."
    scenario_data::DataFrame
    "Wait time in seconds before machines are switched to standby mode."
    wait_before_standby::Int
    "Conversion factor for the energy prices from scenario data (converts prices to €/Ws)."
    price_conversion::Float64
    "Path to export the enery aware production schedule."
    export_schedule_file::String

    function Environment(pyenv::PyObject)
        # Read general settings from the python environment.
        settings = EnvironmentSettings(pyenv)

        # Read the production system structure from configuration files.
        productionsystem =
            ProductionSystem(settings.pathscenarios, pyenv.machines_file, pyenv.products_file, pyenv.orders_file)

        # Create the event and variables mappings.
        events = EventsMaps(productionsystem, pyenv.varsecondmap)

        scenario_data = py_import_scenario(settings, pyenv.scenario_paths; keep_date = true)

        export_schedule_file = hasproperty(pyenv, :export_schedule_file) && isa(pyenv.export_schedule_file, String) ? pyenv.export_schedule_file : "energy_aware_production_schedule.csv"

        # Instantiate the environment
        env = new(
            settings,
            EnvironmentStatus(pyenv),
            productionsystem,
            events,
            Matrix{Float64}(undef, 0, 0),
            Matrix{Int}(undef, 0, 0),
            Matrix{Int}(undef, 0, 0),
            Vector{Vector{Int}}(undef, 0),
            scenario_data,
            pyenv.wait_before_standby,
            pyenv.price_conversion,
            export_schedule_file,
        )

        # Set the action and observation spaces of the python environment.
        action_space = py_dict_space(
            Dict(
                "events" => py_discrete_space(length(env.events.to_schedule)),
                "variables" =>
                    py_multidiscrete_space(fill(length(env.events.varsecondmap), length(env.events.to_schedule))),
            ),
        )
        observation_space = py_box_space(0, numpy.inf, shape=(length(env.events.events),), dtype=numpy.float32)
        pyenv_set_spaces!(env.settings, action_space, observation_space)

        return env
    end

    function Environment(
        settings::EnvironmentSettings,
        status::EnvironmentStatus,
        machines_file::AbstractString,
        products_file::AbstractString,
        orders_file::AbstractString,
        scenario_paths::Vector{Dict{String, T}},
        varsecondmap::Vector{Int},
        wait_before_standby::Number,
        price_conversion::Number,
        export_schedule_file::String = "energy_aware_production_schedule.csv",
    ) where {T <: Any}

        productionsystem = ProductionSystem(settings.pathscenarios, machines_file, products_file, orders_file)
        events = EventsMaps(productionsystem, varsecondmap)
        scenario_data = py_import_scenario(settings, scenario_paths; keep_date = true)

        new(
            settings,
            status,
            productionsystem,
            events,
            Matrix{Float64}(undef, 0, 0),
            Matrix{Int}(undef, 0, 0),
            Matrix{Int}(undef, 0, 0),
            Vector{Vector{Int}}(undef, 0),
            scenario_data,
            wait_before_standby,
            price_conversion,
            export_schedule_file,
        )
    end
end

"""
Perform an environment step.

:param env: The environment

:param actions: Actions as determined by the optimization algorithm.

:return: Tuple of observations, reward, terminated, truncated and info.
"""
function step!(env::Environment, actions)
    env.status.nsteps += 1
    nactions = size(actions)[1]

    if nactions > 1
        observations = Matrix{Float64}(undef, (nactions, length(env.events.events)))
        rewards = Matrix{Float64}(undef, nactions, 3)
        terminated = trues(nactions)
        truncated = falses(nactions)
        infos = [Dict{String, Any}() for _ in 1:nactions]

        # Copy events and variables to make them threadsafe
        events = Matrix{Int}(undef, (nactions, length(env.events.to_schedule)))
        variables = Matrix{Int}(undef, (nactions, length(env.events.to_schedule)))
        for i in 1:nactions
            events[i, :], variables[i, :] = actions[i]
        end

        try
            @threads for i in 1:nactions
                obs, rew, term, trunc, info = calcreward(env, events[i, :], variables[i, :])

                for j in eachindex(obs)
                    if ismissing(obs[j])
                        obs[j] = NaN
                    end
                end
                for j in eachindex(rew)
                    if ismissing(rew[j])
                        rew[j] = NaN
                    end
                end

                observations[i, :], rewards[i, :], terminated[i], truncated[i], infos[i] = obs, rew, term, trunc, info
            end
        catch err
            println("An error occurred: $err")
        end
    else
        events, variables = actions
        observations, rewards, terminated, truncated, infos = calcreward(env, events, variables)
    end

    @debug log_step_errors(infos)

    return observations, rewards, terminated, truncated, infos
end

"""
Update the environment state. Implemented here only for compatibility with the eta_utility interface.

:param env: The environment.
"""
function update!(env::Environment, actions) end

"""
Reset the environment state.

:param env: The environment.
:param seed: The seed that is used to initialize the environment's PRNG.
:param options: Additional information to specify how the environment is reset (optional,
depending on the specific environment) (default: None)

:return: Tuple of observations and info.
"""
function reset!(
    env::Environment,
    seed::Union{Int,Nothing} = nothing,
    options::Union{Dict{String,Any},Nothing} = nothing,
)
    info::Dict{String, Any} = Dict{String, Any}()

    ProductionScheduling.reset!(env.status, env.settings)
    env.buffer_rewards = Matrix{Float64}(undef, 0, 0)
    env.buffer_events = Matrix{Int}(undef, 0, 0)
    env.buffer_variables = Matrix{Int}(undef, 0, 0)
    env.buffer_fronts = Vector{Vector{Int}}(undef, 0)

    observations = zeros(Float64, env.settings.pyenv."observation_space"."shape"[1])

    return observations, info
end

"""
Update the environment state with observations if it interacts with another environment. Not implemented.

:param env: The Environment.
"""
function first_update!(env::Environment) end

"""
Calcuate the reward for a solution that is returned to the agent.

:param env: The environment
:param events: Array of actions as determined by the agent.
:param variables: Array of variables as determined by the agent.

:return: Tuple of observations, reward[makespan, energy costs, max. peak load], terminated, truncated and info.
"""
function calcreward(env::Environment, events, variables)
    schedule, machinegraph, machinestartingnodes, error = buildschedule(env.events, env.structure, events, variables)

    # Store info object
    info = Dict{String, Any}()
    if !isnothing(error)
        info["error"] = error
        info["valid"] = false

        return zeros(Float64, length(env.events.events)), [Inf, Inf, Inf], false, false, info
    end

    # Get the separate schedules for all machines.
    machineschedules = machineschedule(env.events, schedule, machinegraph, machinestartingnodes)

    # Calculate makespan
    makespan = schedule_makespan(machineschedules)

    # If the makespan exceeds the scenario duration, the solution is not allowable because in that case,
    # the energy consumption cannot be calculated.
    if makespan >= env.settings.scenarioduration
        info["error"] = "Exceeded maximum allowable makespan (scenario_duration)."
        info["valid"] = true

        return ones(Float64, length(env.events.events)), [makespan, Inf, Inf], false, false, info
    end

    # Electric power calculated in W -> length of vector is 'makespan'
    electric_power = sum(
        map(
            c -> electric_power_consumption(c),
            values(machineenergy(env, machineschedules, makespan)),
        ))

    # Electric power, difference between the power and the forcast_solar power -> length of vector is 'makespan'
    if hasproperty(env.scenario_data, :forecast_solar)
        electric_power = electric_power .- (env.scenario_data[1:makespan, :forecast_solar] .* 0.5)
    end

    # Set negative power to 0, negative power is not marketed -> length of vector is 'makespan'
    electric_power[electric_power .< 0 ] .= 0

    energy_costs = sum(electric_power .* env.scenario_data[1:makespan, :electrical_energy_price] .* env.price_conversion)

    # Energy consumption calculated in Wh
    # electric_power_con = sum(electric_power) * (1/3600)

    # Calculate the max peak for every machine
    _electric_power = zeros(Float64, length(machineschedules), makespan)
    for (machine, energy) in pairs(machineenergy(env, machineschedules, makespan))
        _electric_power[machine, :] = electric_power_consumption(energy)
    end

    electric_max_peak = maximum(vec(sum(_electric_power, dims=1)))

    return ones(Float64, length(env.events.events)), [makespan, energy_costs, electric_max_peak], false, false, info
end

"""
Calculate the makespan of a production schedule

:param machineschedules: Schedules for all machines.
:return: Makespan
"""
schedule_makespan(machineschedules) = maximum(map(x -> length(x) > 0 ? last(x).endtime : 0, machineschedules))

"""
Calculate the energy consumption of a machine over the course of a production schedule.

:param env: The Environment
:param machineschedules: Schedules for all machines
:param makespan: Total duration of the production schedule.
:return: Vector of vectors of energy consumptions for all machines.
"""
function machineenergy(env::Environment, machineschedules, makespan)
    # Calculate the energy consumption of the solution
    energy_consumption = Vector{ModelResults}(undef, length(machineschedules))

    for (machine, machineschedule) in pairs(machineschedules)

        if typeof(env.structure.machines[machine]) <: LaserCuttingMachine
            a_st2, a_st1, a_op, a_wk = machinestates_lasercutting(env, machineschedule, makespan)
            energy_consumption[machine] = machineenergy_lasercutting(env, machine, a_st2, a_st1, a_op, a_wk)
        elseif typeof(env.structure.machines[machine]) <: SandblastingMachine
            a_st, a_op, a_wk, a_pk = machinestates_sandblasting(env, machineschedule, makespan)
            energy_consumption[machine] = machineenergy_sandblasting(env, machine, a_st, a_op, a_wk, a_pk)
        elseif typeof(env.structure.machines[machine]) <: FlameCuttingMachine
            a_st, a_op, a_wk = machinestates_flamecutting(env, machineschedule, makespan)
            energy_consumption[machine] = machineenergy_flamecutting(env, machine, a_st, a_op, a_wk)
        else
            error("The $(typeof(env.structure.machines[machine])) was not found for calculating the machineenergy.")
        end
    end

    return energy_consumption
end

"""
Calculate the energy consumption of the lasercutting machine over the course of a production schedule.

:param env: The Environment
:param a_st2: Machine state standby2 for the forward model.
:param a_st1: Machine state standby1 for the forward model.
:param a_op: Machine state operational for the forward model.
:param a_wk: Machine state working for the forward model.
:return: Vector of the electric nergy consumption of the lasercutting machine.
"""
function machineenergy_lasercutting(env::Environment, machine, a_st2, a_st1, a_op, a_wk)

    return forward_model(
            env.structure.machines[machine],
            a_st2=a_st2,
            a_st1=a_st1,
            a_op=a_op,
            a_wk=a_wk,
        )
end

"""
Calculate the energy consumption of the flamecutting machine over the course of a production schedule.

:param env: The Environment
:param a_st: Machine state standby2 for the forward model.
:param a_op: Machine state operational for the forward model.
:param a_wk: Machine state working for the forward model.
:return: Vector of the electric nergy consumption of the lasercutting machine.
"""
function machineenergy_flamecutting(env::Environment, machine, a_st, a_op, a_wk)

    return forward_model(
            env.structure.machines[machine],
            a_st=a_st,
            a_op=a_op,
            a_wk=a_wk,
        )
end

"""
Calculate the energy consumption of the sandblasting machine over the course of a production schedule.

:param env: The Environment
:param a_st: Machine state standby2 for the forward model.
:param a_op: Machine state operational for the forward model.
:param a_wk: Machine state working for the forward model.
:param a_pk: Machine state peak for the forward model.
:return: Vector of the electric nergy consumption of the lasercutting machine.
"""
function machineenergy_sandblasting(env::Environment, machine, a_st, a_op, a_wk, a_pk)

    return forward_model(
            env.structure.machines[machine],
            a_st=a_st,
            a_op=a_op,
            a_wk=a_wk,
            a_pk=a_pk,
        )
end

"""
Calculate the states of a laser cutting machine over the course of a production schedule.

:param env: The environment.
:param machineschedule: A schedule for that machine.
:param makespan: Total duration of the production schedule.

:return: The binary variables that specify the states of the production plan.
"""
function machinestates_lasercutting(
    env::Environment,
    machineschedule::Vector{ScheduleItem},
    makespan
)
    a_st2 = falses(makespan)
    a_st1 = falses(makespan)
    a_op = falses(makespan)
    a_wk = falses(makespan)

    previousendtime = 1
    previousjob = Int

    # Duration of the setup before working starts
    wk_previous_setup = 30
    # Starttime of the setup before working starts
    wk_previous_setup_starttime = Int
    # Duration of the setup after working ends
    wk_after_setup = 30
    # Starttime of the setup after working ends
    wk_after_setup_starttime = Int

    # Starttime of working
    wk_starttime = Int

    for item in machineschedule

        # Asumption for the calculation of the starttime: starttime < duration <= endtime
        wk_previous_setup_starttime = item.starttime + item.setuptime
        wk_starttime = wk_previous_setup_starttime + wk_previous_setup
        wk_after_setup_starttime = item.endtime - wk_after_setup

        # First working mode on the machine
        if previousendtime == 1
            a_op[wk_previous_setup_starttime+1:wk_starttime] .= true
            a_wk[wk_starttime+1:wk_after_setup_starttime] .= true
            a_op[wk_after_setup_starttime+1:item.endtime] .= true

            previousjob = item.job
            previousendtime = item.endtime
            continue
        end

        if wk_starttime - previousendtime >= env.wait_before_standby && previousendtime != 1
            a_st2[previousendtime+1:wk_starttime] .= true
        elseif (previousendtime != wk_starttime)
            a_op[previousendtime+1:wk_starttime] .= true
        end

        a_op[wk_previous_setup_starttime+1:wk_starttime] .= true
        a_wk[wk_starttime+1:wk_after_setup_starttime] .= true
        a_op[wk_after_setup_starttime+1:item.endtime] .= true

        previousendtime = item.endtime
        previousjob = item.job
    end

    return a_st2, a_st1, a_op, a_wk
end

"""
Calculate the states of a flamecutting machine over the course of a production schedule.

:param env: The environment.
:param machineschedule: A schedule for that machine.
:param makespan: Total duration of the production schedule.

:return: The binary variables that specify the states of the production plan.
"""
function machinestates_flamecutting(
    env::Environment,
    machineschedule::Vector{ScheduleItem},
    makespan
)

    a_st = falses(makespan)
    a_op = falses(makespan)
    a_wk = falses(makespan)

    previousendtime = 1
    previousjob = Int

    # Starttime of working
    wk_starttime = Int

    for item in machineschedule

        # Asumption for the calculation of the starttime: starttime < duration <= endtime
        wk_starttime = item.starttime + item.setuptime

        # First working mode on the machine
        if previousendtime == 1
            a_wk[wk_starttime+1:item.endtime] .= true

            previousjob = item.job
            previousendtime = item.endtime
            continue
        end

        if wk_starttime - previousendtime >= env.wait_before_standby && previousendtime != 1
            a_st[previousendtime+1:wk_starttime] .= true
        elseif previousendtime != wk_starttime
            a_op[previousendtime+1:wk_starttime] .= true
        end

        a_wk[wk_starttime+1:item.endtime] .= true

        previousendtime = item.endtime
        previousjob = item.job
    end

    return a_st, a_op, a_wk
end

"""
Calculate the states of a sandblasting machine over the course of a production schedule.

:param env: The environment.
:param machineschedule: A schedule for that machine.
:param makespan: Total duration of the production schedule.

:return: The binary variables that specify the states of the production plan.
"""
function machinestates_sandblasting(
    env::Environment,
    machineschedule::Vector{ScheduleItem},
    makespan
)
    a_st = falses(makespan)
    a_op = falses(makespan)
    a_wk = falses(makespan)
    a_pk = falses(makespan)

    previousendtime = 1
    previousjob = Int

    # Duration peak
    pk_duration = 5
    # Starttime peak
    pk_starttime = Int
    # Starttime of working
    wk_starttime = Int

    for item in machineschedule

        # Asumption for the calculation of the starttime: starttime < duration <= endtime
        wk_starttime = item.starttime + item.setuptime

        # First working mode on the machine
        if previousendtime == 1
            pk_duration_opt = pk_duration
            pk_starttime = wk_starttime + pk_duration_opt

            a_pk[wk_starttime+1:pk_starttime] .= true
            a_wk[pk_starttime+1:item.endtime] .= true

            previousjob = item.job
            previousendtime = item.endtime
            continue
        end

        if wk_starttime - previousendtime >= env.wait_before_standby && previousendtime != 1
            a_st[previousendtime:wk_starttime] .= true
        elseif previousendtime != wk_starttime
            a_op[previousendtime:wk_starttime] .= true
        end

        # Peak exists, when machine starts from off or is in mode standby
        condition = a_st[item.starttime] == true
        # Peak time depends on the mode
        pk_duration_opt = condition ? pk_duration : 0
        pk_starttime = wk_starttime + pk_duration_opt

        a_pk[wk_starttime+1:pk_starttime] .= condition ? true : false
        a_wk[pk_starttime+1:item.endtime] .= true


        previousendtime = item.endtime
        previousjob = item.job
    end

    return a_st, a_op, a_wk, a_pk
end

"""
Calculate the total electric energy consumption of a lasercutting machine from the energy model results object.

:param data: Energy model results.
:return: Timeseries of electric energy consumption.
"""
electric_power_consumption(data::LaserCuttingMachineResults) =
    data.P_el

"""
Calculate the total electric energy consumption of a sandblasting machine from the energy model results object.

:param data: Energy model results.
:return: Timeseries of electric energy consumption.
"""
electric_power_consumption(data::SandblastingMachineResults) =
    data.P_el

"""
Calculate the total electric energy consumption of a flamecutting machine from the energy model results object.

:param data: Energy model results.
:return: Timeseries of electric energy consumption.
"""
electric_power_consumption(data::FlameCuttingMachineResults) =
    data.P_el

"""
Render the optimization results in a scatter plot, a Gannt chart and some energy plots.

:param env: The environment.
"""
function render(env::Environment; kwargs...)

    kwargs = isa(kwargs, Base.Pairs) ? Dict(kwargs) : Dict(kwargs...)

    mode = haskey(kwargs, :mode) ? kwargs[:mode] : "human"
    path = haskey(kwargs, :path) ? kwargs[:path] : env.settings.pathresults
    filename = haskey(kwargs, :filename) ? kwargs[:filename] : nothing
    fileextension = haskey(kwargs, :fileextension) ? kwargs[:fileextension] : "png"
    debug_annotations = haskey(kwargs, :debug_annotations) ? kwargs[:debug_annotations] : true

    erc = Vector{Tuple{Int, Float64, Float64}}()

    if isnothing(filename)
        env.status.nepisodes += 1
        filename=etautility.eta_x.common.episode_name_string(env.settings.runname, env.status.nepisodes, env.settings.envid)
    end

    solution_path = joinpath(path, "$(filename).json")

    if isfile(solution_path)
        # File to save the solution already exists, read the content
        json = JSON.parsefile(solution_path; use_mmap=false)
    else
        # File to save the solution doesn't exist, create an empty array
        json = []
    end

    plotargs = Dict(
        :plot_titlefontsize => 14,
        :annotation_fontsize => 14,
        :guidefont => 14,
        :ytickfont => 14,
        :ylabelfontsize => 14,
        :xlabelfontsize => 14,
        :xtickfont => 14,
        :legendfont => 14,
        :xguidefont => 14,
        :yguidefont => 14,
        :formatter => :plain,
        :fontfamily => "Helvetica",
        :margin => (1, :mm),
        :top_margin => (1, :mm),
        :bottom_margin => (2, :mm),
        :right_margin => (7, :mm),
        :left_margin => (7, :mm),
        :legend_background_color => RGBA(1, 1, 1, 0.7),
        :size => (800, 900),
        :dpi => 150,
        :size => (
            1.75 * 321.51616 * 120 * 0.01384, # width in pt * dpi * inches per pt
            1.75 * 469.4704 * 120 * 0.01384 * 0.95, # heigth in pt * dpi * inches per pt * 0.95
        ),
    )

    if typeof(env.buffer_fronts) <: Matrix
        env.buffer_fronts = [env.buffer_fronts[i, :] for i in 1:size(env.buffer_fronts)[1]]
    end

    if length(env.buffer_rewards) > 0
        thisplotargs = copy(plotargs)
        thisplotargs[:size] = (first(plotargs[:size]) * 0.7, last(plotargs[:size]) * 0.3)
        plot_solspace = render_solspace(env, debug_annotations; thisplotargs...)
        savefig(plot_solspace, joinpath(path, "$(filename)_solutionspace.$fileextension"))

        if mode == "all"
            for front in env.buffer_fronts, solution in front .+ 1
                electric_power_consumption = render_solution(
                    env,
                    solution,
                    path,
                    filename,
                    fileextension,
                    solution_path,
                    json;
                    plotargs...
                )
                # Save energy related costs (erc) and solution number in vector in vector to export the production
                # scheduling plan with the minimal erc
                push!(erc, (solution, env.buffer_rewards[solution, 2], electric_power_consumption))
            end

        else
            solution = typeof(mode) <: String ? 1 : mode
            electric_power_consumption = render_solution(
                env,
                solution,
                path,
                filename,
                fileextension,
                solution_path,
                json;
                plotargs...
            )
        end
    end

    min_erc = !isempty(erc) ? argmin(t -> t[2], erc) : nothing

    if !isnothing(min_erc)
        export_production_scheduling(env, min_erc; dates_format=true)
    end
end

"""
Render the solution space determined by the agent.

:param env: The environment.
:param debug_annotations:

:return: The solution space as plot.
"""
function render_solspace(
    env::Environment,
    debug_annotations::Bool;
    kwargs...
)

    _comb = [("MSKP","ERC"), ("MKSP","PK"), ("ERC","PK")]
    _fronts = Dict("MSKP" => 1, "ERC" => 2, "PK" => 3)
    _labels = Dict("MSKP" => "Makespan in s", "ERC" => "Energy costs in EUR", "PPC" => "Max. Peak in W")

    maxfront = 8
    plt = plot(; xlabel="Makespan in s", ylabel="Energy costs in EUR", palette=:okabe_ito, kwargs...)

    for (num, front) in enumerate(env.buffer_fronts)
        # Only show the first eight fronts
        num > maxfront && break
        scatter!(plt, env.buffer_rewards[front.+1, 1], env.buffer_rewards[front.+1, 2], label="front $num")
    end

    if debug_annotations
        for (num, front) in enumerate(env.buffer_fronts)
            # Only show the first eight fronts
            num > maxfront && break
            # Plot only annotations, moved up slightly compared to the actual markers.
            scatter!(
                plt,
                env.buffer_rewards[front.+1, 1],
                env.buffer_rewards[front.+1, 2] .+ ((last(ylims(plt)) - first(ylims(plt))) / 60),
                markeralpha=0,
                color=:lightgrey,
                label=false,
                series_annotations=text.(front .+ 1, :outerbottom, 7, "Helvetica"),
            )
        end
    end

    return plt
end

"""
Render a specific schedule that includes a gant chart, electrical power and the energy costs.

:param env: The environment.
:param solutionidx: Index of the solution to be rendered.

:return: Plots of the production plan as a gant chart, the resulting energy output and the corresponding energy costs.
"""
function render_schedule(
    env::Environment,
    solutionidx::Int;
    kwargs...
)

    schedule, machinegraph, machinestartingnodes, err = buildschedule(
        env.events,
        env.structure,
        env.buffer_events[solutionidx, :],
        env.buffer_variables[solutionidx, :];
        debug=true
    )
    if !isnothing(err)
        errtext = "The requested solution '$solutionidx' is not valid and cannot be rendered: $err"
        @error errtext

        plt = plot(; kwargs...)
        annotate!(plt, [(0.5, 0.5, (errtext, 11, :red, :center))])
        return plt
    end

    # Set fontsize for every plots
    _fontsize = 14

    machineschedules = machineschedule(env.events, schedule, machinegraph, machinestartingnodes)
    makespan = schedule_makespan(machineschedules)

    #  Specify the date length of the experiment data, date_length must not be greater than the experiment solution.
    date_length = makespan <= size(env.scenario_data)[1] ? makespan : size(env.scenario_data)[1]

    ################################################
    # Plotting of the scheduling for every machine #
    ################################################

    plot_schedule, yticklabels = render_schedule_machine(env, machineschedules, makespan)

    # Formatting the y-axis
    yaxis!(
        plot_schedule,
        ylims=(0.4, length(yticklabels) + 1.4 + 2*(0.2 * length(yticklabels))),
        yticks=(1:length(yticklabels), yticklabels)
    )

    #########################################################################################################
    # Plot the electrical power consumption of the machine schedule, the energy costs and the energy tarifs #
    #########################################################################################################

    if makespan <= size(env.scenario_data)[1]

        plot_power, electric_power = render_power(env, machineschedules, makespan; fontsize=_fontsize)

        plot_prices = render_energy_costs(env, makespan, electric_power; fontsize=_fontsize)
    else
        # Electrical power consumption
        plot_power = plot(
        guidefont=font(_fontsize),
        ylabel="Electric power in kW",
        color=:black,
        legend=false
        )

        annotate!(
            plot_power,
            [(0.5, 0.5, ("could not plot energy, scenario data vector too short.", 11, :red, :center))],
        )
        plot_prices = plot(
            ylabel="Electric energy price in EUR/MWh",
            color=:black,
            legend_position=:topleft,
        )
    end

    plots = [plot_schedule, plot_power, plot_prices]

    for plot in plots
        # Formatting the y-axis
        yaxis!(plot, ytickfont=font(6), yfontfamily="Helvetica")
        # Formatting the x-axis
        format_xaxis(env, plot, date_length; fontsize=_fontsize)
    end

    electric_power_consumption = @isdefined(electric_power) ? (sum(sum(electric_power, dims=1))*(1/3600)) : nothing

    return plot(plot_schedule, plot_power, plot_prices; layout=@layout[°; °; °], kwargs...), electric_power_consumption
end

"""
Helper function for formatting the X-axis with the time stamp of the experiment data.

:param env: The environment.
:param plot: Plot to reformatting the x axis.
:param date_length: Length of the data vector for the graphical display. Normally, date_length is equal to makespan.
:param xtickfont: Set the font size of the date on the x-axis.
"""
function format_xaxis(
    env::Environment,
    plot::Union{Plots.Plot, Plots.Subplot},
    date_length::Int64;
    fontsize = 14
)

    # Extract first and last timestamp
    first_timestamp = first(env.scenario_data[1:date_length, :date])
    last_timestamp = last(env.scenario_data[1:date_length, :date])

    # Create X-ticks based on the timestamps
    xticks = range(first_timestamp, last_timestamp, step=Hour(2))
    indices = findall(x -> x in xticks, env.scenario_data[1:date_length, :date])

    # Set X-Ticks format
    xticklabels = Dates.format.(xticks, "HH:MM")

    xaxis!(
        plot,
        xticks=(indices, xticklabels),
        xtickfont=font(fontsize),
        guidefont=font(fontsize),
        xfontfamily="Helvetica",
        xlabel="Time in HH:MM",
        xtickcolor=:gray,
        xrotation = 45
    )

end

"""
Function to render the schedule, export and print the solution.

:param env: The environment.
:param solution: Index of the solution to be rendered, exported and printered.
:param path: Path for the 'results' folder.
:param filename: Filename for the solutions.
:param fileextension: Extension (i.e. 'svg' or 'png') for the plots.
:param solution_path: Path to store the results.
:param json: Create file to save the results in an extra file.
"""
function render_solution(
    env::Environment,
    solution::Union{Int, String},
    path::String,
    filename::String,
    fileextension::String,
    solution_path::String,
    json::Vector{Any};
    kwargs...
)
    # Render the solution schedule and save the plot
    plot_schedule, electric_power_consumption = render_schedule(env, solution; kwargs...)
    savefig(plot_schedule, joinpath(path, "$(filename)_$solution.$fileextension"))

    # Prepare the solutions of the MKSP, ERC, PPC to save them in a json file
    append!(json, Dict("$(filename)_$solution" => Dict(
        "MKSP" => "$(env.buffer_rewards[solution, 1]) s",
        "ERC" => "$(env.buffer_rewards[solution, 2]) €",
        "PPC" => "$(env.buffer_rewards[solution, 3]/1000) kW",
        "EC" => "$(electric_power_consumption) Wh"
    )))

    # Export solution data as file
    open(normpath(solution_path), "w") do f
        JSON.print(f, json, 4)
    end

    return electric_power_consumption
end

"""
Render the production schedule as gant chart for every machine.

:param env: The environment.
:param machineschedules: Vector of schedules for each machine.
:param makespan: Total duration of the production schedule.

:return: Plot of the energy consumption and the forcestast.solar power is optionally included,
        electrical power of the production schedule as vector.
"""
function render_power(
    env::Environment,
    machineschedules::Vector{Vector{ScheduleItem}},
    makespan::Int;
    fontsize=14
)

    electric_power = zeros(Float64, length(machineschedules), makespan)
    for (machine, energy) in pairs(machineenergy(env, machineschedules, makespan))
        electric_power[machine, :] = electric_power_consumption(energy)
    end

    # Optionally plot the predicted solar power.
    if hasproperty(env.scenario_data, :forecast_solar)
        # Electrical power of the production schedule
        plot(
            vec(sum(electric_power, dims=1) .* 1/1000),
            guidefont=font(fontsize),
            ylabel="Electric power in kW",
            label="elec. power",
            legend=(0.5, 0.9),
            color=:black,
            linecolor=:black
        )

        # Electrical forecast.solar power
        plot_power = plot!(
            twinx(),
            env.scenario_data[1:makespan, :forecast_solar] * 1/1000,
            guidefont=font(fontsize),
            yaxis="Electric power forecast.solar in kW",
            label="elec. power forcast.solar",
            legend=(0.5, 0.8),
            linestyle=:dash,
            linecolor=:black
        )
    else
        # Electrical power of the production schedule
        plot_power = plot(
            vec(sum(electric_power, dims=1) .* 1/1000),
            linecolor=:black,
            guidefont=font(fontsize),
            ylabel="Electric power in kW",
            color=:black,
            legend=false
        )
    end

    return plot_power, electric_power
end

"""
Render the production schedule as gant chart for every machine.

:param env: The environment.
:param machineschedules: Vector of schedules for each machine.
:param makespan: Total duration of the production schedule.

:return: The gant chart of the production schedule for every machine and the yticklabels of the plot.
"""
function render_schedule_machine(
    env::Environment,
    machineschedules::Vector{Vector{ScheduleItem}},
    makespan::Int
)

    plot_schedule = plot(legend=:top, legend_columns=4)
    # plot_schedule = plot(legend_columns=3)
    yticklabels = String[]

    # Setup legend
    bar!(plot_schedule, (-10, -10), orientation=:h, color=palette(:Greys_8)[end], label="Pause Time")
    bar!(plot_schedule, (-10, -10), orientation=:h, color=palette(:Greys_8)[3], label="Setup Time")
    labels = falses(length(env.structure.products))

    for (machine, schedule) in pairs(machineschedules)
        push!(yticklabels, env.structure.machines[machine].name)
        starts = Int[]
        texts = String[]
        for item in schedule
            # Preceding pause
            if item.starttime - item.pausetime < item.starttime
                bar!(
                    plot_schedule,
                    (machine, item.starttime),
                    fillrange=item.starttime - item.pausetime,
                    bar_width=0.2,
                    orientation=:h,
                    color=palette(:Greys_8)[end],
                    label=false
                )
            end

            # Actual operation
            bar!(
                plot_schedule,
                (machine, item.endtime),
                fillrange=item.starttime + item.setuptime,
                bar_width=0.7,
                orientation=:h,
                color=palette(:Paired_12)[item.product],
                label=labels[item.product] ? false : env.structure.products[item.product].name
            )
            labels[item.product] = true
            if length(item.coinciding) > 1
                push!(starts, item.starttime + item.setuptime)
                push!(texts, "$(length(item.coinciding))")
            end

            # Smaller line for preparation time
            bar!(
                plot_schedule,
                (machine, item.starttime + item.setuptime),
                fillrange=item.starttime,
                bar_width=0.2,
                orientation=:h,
                color=palette(:Greys_8)[3],
                label=false
            )
        end

        if !isempty(starts)
            annotate!(
                plot_schedule,
                starts .+ 0.005 * makespan,
                machine + 0.32,
                text.(texts, :left, :top, 14, "Helvetica"),
            )
        end
    end

    return plot_schedule, yticklabels
end

"""
Render the energy tarifs and the energy costs as graph.

:param env: The environment.
:param machineschedules: Vector of schedules for each machine.
:param makespan: Total duration of the production schedule.

:return: Plot of the energy tarifs and the energy costs.
"""
function render_energy_costs(
    env::Environment,
    makespan::Int,
    electric_power::Matrix{Float64};
    fontsize = 12
    )

    plot(
        env.scenario_data[1:makespan, :electrical_energy_price],
        guidefont=font(fontsize),
        yaxis="Electric energy price in EUR/MWh",
        label="energy price",
        legend=(0.1, 0.9),
        color=:black,
        linecolor=:black
    )

    plot_prices = plot!(
        twinx(),
        cumsum(
            sum(electric_power, dims=1)[1, :] .* env.scenario_data[1:makespan, :electrical_energy_price] .*
            env.price_conversion,
        ),
        guidefont=font(fontsize),
        yaxis="Cum. electric energy costs in EUR",
        label="cum. elec. energy costs",
        legend=(0.1, 0.7),
        linestyle=:dash,
        linecolor=:black
    )
    return plot_prices
end

"""
Export the production scheduling plan.

:param env: The environment.
:param solution: Index of the solution to be exported.
"""
function export_production_scheduling(
    env::Environment,
    solution::Tuple{Int, Float64, Float64};
    dates_format=false
)

    println("The minimal ERC are $(solution[2])€ for solution $(solution[1])")
    println("The corresponding EC is $(solution[3])W")

    filepath = joinpath(env.settings.pathresults, env.export_schedule_file)
    filepath_power = joinpath(env.settings.pathresults, "schedule_power.csv")

    df = DataFrame(starttime = Int[], working_starttime = Int[], endtime = Int[], machine = Int[], job = Int[])

    schedule, machinegraph, machinestartingnodes, _ = buildschedule(
        env.events,
        env.structure,
        env.buffer_events[solution[1], :],
        env.buffer_variables[solution[1], :];
        debug=true
    )

    machineschedules = machineschedule(env.events, schedule, machinegraph, machinestartingnodes)
    makespan = schedule_makespan(machineschedules)

    electric_power = zeros(Float64, length(machineschedules), makespan)
    for (machine, energy) in pairs(machineenergy(env, machineschedules, makespan))
        electric_power[machine, :] = electric_power_consumption(energy)
    end

    electric_power = electric_power'
    electric_power_date = env.scenario_data[1:makespan, :date]

    for (machine, schedule) in pairs(machineschedules)
        for item in schedule

            if dates_format == true
                _starttime = env.scenario_data[item.starttime+1, :date]
                _working_starttime = env.scenario_data[(item.starttime + 1 + item.setuptime), :date]
                _endtime = env.scenario_data[item.endtime, :date]
            else
                _starttime = item.starttime
                _working_starttime = (item.starttime + item.setuptime)
                _endtime = item.endtime
            end

            df = vcat(
                df,
                DataFrame(
                    starttime = _starttime,
                    working_starttime = _working_starttime,
                    endtime = _endtime,
                    machine = env.structure.machines[machine].id,
                    job = env.structure.products[item.product].id)
                )
        end
    end

    # add new column 'capacity' and initially fill with 1
    df.capacity .= fill(1, nrow(df))
    cap = 1

    sort!(df, :starttime)

    # mark duplicates in the 'capacity' column, set the 'capacity' column with the frequency of the duplicate
    duplicates = nonunique(df)
    for (idx, val) in enumerate(duplicates)
        if val == 1
            cap += 1
        else
            if cap > 1 && idx > cap
                df.capacity[idx-cap] = cap
            end
            df.capacity[idx] = 1
            cap = 1
        end
    end

    # delete duplicates row
    df = df[.!duplicates, :]

    sort!(df, :starttime)

    # check whether the directory exists and create it if not
    if !isdir(dirname(filepath))
        mkpath(dirname(filepath))
    end

    # check whether the directory exists and create it if not
    if !isdir(dirname(filepath_power))
        mkpath(dirname(filepath_power))
    end

    CSV.write(filepath, df)

    df_new = DataFrame(hcat(electric_power, electric_power_date), :auto)
    CSV.write(filepath_power, df_new)

end

"""
Close the environment after the optimization is done.

:param env: The environment.
"""
function close!(env::Environment) end

generate_expert_schedules(env::Environment, method::String, count::Int=1) =
    generate_expert_schedules(env.settings, env.events, env.structure, method, count)
