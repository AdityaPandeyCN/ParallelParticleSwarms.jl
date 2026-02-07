@kernel function simplebfgs_run!(nlprob, x0s, result, opt, maxiters, abstol, reltol)
    i = @index(Global, Linear)
    if i <= length(x0s)
        nlcache = remake(nlprob; u0 = x0s[i])
        sol = solve(nlcache, opt; maxiters, abstol, reltol)
        @inbounds result[i] = sol.u
    end
end

function SciMLBase.solve!(
        cache::HybridPSOCache, opt::HybridPSO{Backend, LocalOpt}, args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 100, local_maxiters = 10, kwargs...
    ) where {
        Backend, LocalOpt <: Union{LBFGS, BFGS},
    }
    pso_cache = cache.pso_cache

    sol_pso = solve!(pso_cache)
    x0s = sol_pso.original

    backend = opt.backend

    prob = remake(cache.prob, lb = nothing, ub = nothing)

    result = cache.start_points
    copyto!(result, x0s)

    ∇f = instantiate_gradient(prob.f.f, prob.f.adtype)

    nlprob = SimpleNonlinearSolve.ImmutableNonlinearProblem{false}(∇f, prob.u0, prob.p)

    nlalg = opt.local_opt isa LBFGS ?
        SimpleLimitedMemoryBroyden(;
            threshold = opt.local_opt.threshold,
            linesearch = Val(true)
        ) : SimpleBroyden(; linesearch = Val(true))

    t0 = time()

    x0s_cpu = Array(x0s)
    result_cpu = Array(result)
    for i in eachindex(x0s_cpu)
        nlcache = remake(nlprob; u0 = x0s_cpu[i])
        sol = SimpleNonlinearSolve.solve(nlcache, nlalg; maxiters = local_maxiters, abstol, reltol)
        result_cpu[i] = sol.u
    end
    copyto!(result, result_cpu)

    raw_f = prob.f.f
    p = prob.p
    sol_bfgs = (x -> raw_f(x, p)).(result)
    sol_bfgs = (x -> isnan(x) ? convert(eltype(prob.u0), Inf) : x).(sol_bfgs)

    minobj, ind = findmin(sol_bfgs)
    sol_u,
        sol_obj = minobj > sol_pso.objective ? (sol_pso.u, sol_pso.objective) :
        (view(result, ind), minobj)
    t1 = time()

    solve_time = (t1 - t0) + sol_pso.stats.time

    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        sol_u, sol_obj,
        stats = Optimization.OptimizationStats(; time = solve_time)
    )
end
