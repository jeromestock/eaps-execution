import Dates
using PyCall
import Base.Filesystem: abspath
import StatsBase
import Graphs: nv
using GraphPlot
using Compose
import Cairo, Fontconfig

include("../trials/test/ETAFactoryEnv.jl")

function test()
    env = Environment(
        EnvironmentSettings(
            "trials/test/results",
            "trials/test/scenario",
            1,
            "run1",
            25200,
            1,
            Dates.DateTime("29.11.2021 08:00", "dd.mm.yyyy HH:MM"),
            Dates.DateTime("29.11.2021 10:00", "dd.mm.yyyy HH:MM"),
        ),
        EnvironmentStatus(),
        "productionsystem/machines.json",
        "productionsystem/products.json",
        "orders.json",
        [
            Dict("path" => "../../scenario/Strompreise_211129.csv", "interpolation_method" => "ffill", "time_conversion_str" => "ISO8601", "date_format" => "%Y-%m-%d %H:%M:%S")
            Dict("path" => "../../scenario/ambient_temperatures.csv", "time_conversion_str" => "ISO8601", "date_format" => "%Y-%m-%d %H:%M:%S")
        ],
        [0, 180, 360],
        0.8,
        0.95,
        180,
        Dict("971" => 20.0, "972" => 20.0, "985" => 60.0, "981" => 60.0),
        0.0000000002777,
    )

    # events = Random.shuffle(convert(Vector{Int}, 0:22))
    # events = [6, 7, 1, 2, 4, 9, 3, 5, 8, 0]
    # events = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    events = [8, 2, 3, 4, 10, 1, 0, 7, 6, 9, 5, 11]
    # events = [8, 2, 13, 3, 4, 10, 14, 1, 0, 7, 17, 6, 16, 12, 19, 9, 5, 15, 18, 11]
    # events = [10, 13, 15, 17, 4, 9, 16, 7, 18, 12, 1, 20, 14, 6, 11, 5, 19, 21, 8, 3, 0, 2]
    # events = [31, 29, 18,  5, 19, 33,  3, 16, 38, 53,  0,  1, 13, 15, 39, 21, 34, 46, 25, 57, 48, 30, 44, 49, 54,  2, 43, 17,  8, 51, 40, 10,  7, 36, 55, 32, 14, 23, 37, 27,  9, 45,  6, 52, 22, 58, 20, 24,  4, 47, 28, 41, 26, 59, 42, 12, 11, 56, 50, 35]

    # variables = StatsBase.sample(0:4, 23, replace=true)
    # variables = [0, 2, 0, 2, 2, 0, 0, 0, 2, 1]#
    variables = [2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 1, 1, 2, 0, 1, 2, 2, 1, 1, 0]
    # variables = [1, 1, 2, 2, 1, 2, 1, 0, 2, 0, 1, 2, 0, 1, 2, 1, 1, 2, 1, 1, 0, 2]
    # variables = [0, 1, 0, 2, 2, 2, 0, 2, 1, 1, 0, 0, 2, 1, 2, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 2, 2, 1, 0, 2, 1, 2, 1, 2, 2, 1, 1, 2, 0, 1, 1, 2, 1, 2, 0, 1, 2, 0, 0, 0, 2, 1, 2, 2, 1, 0, 1, 2, 2, 2]

    # scheduled_nodes = [1,2,3,4,5,6,9,10]

    # observations, rewards, dones, infos = step!(env, [vec for vec in [[events, variables], [events, variables]]])
    # observations, rewards, dones, infos = calcreward(env, events, variables)
    schedule, machinegraph, machinestartingnodes, error = buildschedule(env.events, env.structure, events, variables, debug=true)

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
