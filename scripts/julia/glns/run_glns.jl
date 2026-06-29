#!/usr/bin/env julia

using GLNS
using Random

if length(ARGS) != 5
    println(stderr, "usage: run_glns.jl INSTANCE OUTPUT MODE MAX_TIME_SECONDS SEED")
    exit(2)
end

instance, output, mode = ARGS[1], ARGS[2], ARGS[3]
max_time = parse(Int64, ARGS[4])
seed = parse(Int64, ARGS[5])

Random.seed!(seed)
GLNS.solver(
    instance;
    output=output,
    mode=mode,
    max_time=max_time,
    verbose=0,
)
