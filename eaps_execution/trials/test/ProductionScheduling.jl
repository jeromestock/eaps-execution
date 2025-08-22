import Dates
using PyCall
import Base.Filesystem: abspath
import StatsBase
import Graphs: nv
using GraphPlot
using Compose
import Cairo, Fontconfig

include("../environments/FactoryEnv.jl")

function test()
env = Environment(
        EnvironmentSettings(
            joinpath(@__DIR__, "../results"),
            joinpath(@__DIR__, "../scenario"),
            1,
            "run1",
            25200,
            1,
            Dates.DateTime("22.04.2024 08:00", "dd.mm.yyyy HH:MM"),
            Dates.DateTime("22.04.2024 16:00", "dd.mm.yyyy HH:MM"),
        ),
        EnvironmentStatus(),
        joinpath("productionsystem", "machines.json"),
        joinpath("productionsystem", "products.json"),
        "orders_prod.json",
        [
            Dict("path" => "../../scenario/Strompreise_240422.csv", "interpolation_method" => "ffill", "time_conversion_str" => "mixed"),
            Dict("path" => "../../scenario/forecast_solar.csv", "time_conversion_str" => "mixed")
        ],
        [0, 180, 300, 600, 900, 1800, 3600],
        180,
        0.0000000002777,
        # [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 19, 20, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44]
        # [1, 2, 4, 5, 37, 38, 39, 40, 44],
        # [1, 2, 4, 5, 24, 28, 30, 31, 32, 37, 38, 39, 40, 44]
    )

    events = Random.shuffle(convert(Vector{Int}, 0:(length(env.events.to_schedule)-1)))
    # events = [6, 7, 1, 2, 4, 9, 3, 5, 8, 0]
    # events = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # events = [8, 2, 3, 4, 10, 1, 0, 7, 6, 9, 5, 11]
    # events = [8, 2, 13, 3, 4, 10, 14, 1, 0, 7, 17, 6, 16, 12, 19, 9, 5, 15, 18, 11]
    # events = [10, 13, 15, 17, 4, 9, 16, 7, 18, 12, 1, 20, 14, 6, 11, 5, 19, 21, 8, 3, 0, 2]
    # events = [31, 29, 18,  5, 19, 33,  3, 16, 38, 53,  0,  1, 13, 15, 39, 21, 34, 46, 25, 57, 48, 30, 44, 49, 54,  2, 43, 17,  8, 51, 40, 10,  7, 36, 55, 32, 14, 23, 37, 27,  9, 45,  6, 52, 22, 58, 20, 24,  4, 47, 28, 41, 26, 59, 42, 12, 11, 56, 50, 35]

    # variables = StatsBase.sample(0:4, 23, replace=true)
    # variables = [0, 2, 0, 2, 2, 0, 0, 0, 2, 1]#
    variables = [2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 1, 1, 2, 0, 1, 2, 2, 1, 1, 0, 2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 1, 1, 2, 0, 1, 2, 2, 1, 1, 0, 1, 0]
    # variables = [2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 1, 1, 2, 0, 1, 2, 2, 1, 1, 0, 2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 1, 1, 2, 0, 1, 2, 2, 1, 1, 0, 1]
    # variables = [2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 1, 1, 2, 0, 1, 2, 2, 1, 1, 0, 2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 1, 1, 2, 0, 1, 2, 2, 2, 1]
    # variables = [1, 1, 2, 2, 1, 2, 1, 0, 2, 0, 1, 2, 0, 1, 2, 1, 1, 2, 1, 1, 0, 2]
    # events_not_schedule
    # variables = [1, 1, 2, 2, 1, 2, 1, 0, 2, 0, 1, 2, 0, 1, 2]
    # variables = [2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 1, 1, 2, 0, 1, 2, 2, 1, 1, 0, 2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 1, 1, 2, 0, 1]
    # variables = [2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 1, 1, 2, 0, 1, 2, 2, 1, 1, 0, 2, 0, 2, 1, 0, 2, 0, 2, 2, 1]


    # variables = [0, 1, 0, 2, 2, 2, 0, 2, 1, 1, 0, 0, 2, 1, 2, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 2, 2, 1, 0, 2, 1, 2, 1, 2, 2, 1, 1, 2, 0, 1, 1, 2, 1, 2, 0, 1, 2, 0, 0, 0, 2, 1, 2, 2, 1, 0, 1, 2, 2, 2]

    # scheduled_nodes = [1,2,3,4,5,6,9,10]

    # observations, rewards, dones, infos = step!(env, [vec for vec in [[events, variables], [events, variables]]])
    observations, rewards, dones, infos = calcreward(env, events, variables)
    schedule, machinegraph, machinestartingnodes, error = buildschedule(env.events, env.structure, events, variables, debug=true)

    machineschedules = machineschedule(env.events, schedule, machinegraph, machinestartingnodes)

    for (machine, schedule) in pairs(machineschedules)
        println(machine)
        for item in schedule
            println(item)
        end
    end

    events, events_to_schedule, events_in_storage, productgraph = mapevents(env.structure; debug=true)

    println(machinestartingnodes)
    # println(events .+ 1)
    println(error)
    # println(schedule, machinestartingnodes, error)

    # draw(
    #     PNG("productgraph.png", 16cm, 16cm),
    #     gplot(env.productgraph, layout=shell_layout, nodelabel=1:nv(env.productgraph), arrowlengthfrac=0.05),
    # )

    # draw(
    #     PNG("machinegraph.png", 16cm, 16cm),
    #     GraphPlot.gplot(machinegraph, layout=spring_layout, nodelabel=1:nv(machinegraph), arrowlengthfrac=0.05),
    # )

    # retries = 1
    # for r in 1:retries
    #     events = StatsBase.sample(0:9, 10, replace=false)
    #     schedule, machinestartingnodes, error = buildschedule(env, events, variables)
    #     println("retries: ", r)

    #     if isnothing(error)
    #         break
    #     else
    #         continue
    #     end
    #     println(events)
    #     println(schedule)
    # end

    output = generate_expert_schedules(env.settings, env.events, env.structure, "spt", 1)
    println(output)
end
