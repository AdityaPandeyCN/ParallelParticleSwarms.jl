using KernelAbstractions
using SciMLBase
using Optimization
using LineSearch
using NonlinearSolveQuasiNewton

@kernel function simplebfgs_run!(nlprob, x0s, result, opt, maxiters, abstol, reltol, grad_f)
    i = @index(Global, Linear)
    nlcache = SciMLBase.init(
        nlprob, opt; u0 = x0s[i], maxiters, abstol, reltol, grad_f,
        kwargshandle = SciMLBase.KeywordArgSilent
    )
    sol = SciMLBase.solve!(nlcache)
    @inbounds result[i] = sol.u
end

# HybridPSO: global PSO exploration, then per-particle local BFGS refinement.
function SciMLBase.solve!(
        cache::HybridPSOCache, opt::HybridPSO{Backend, LocalOpt}, args...;
        abstol = nothing, reltol = nothing, maxiters = 100, local_maxiters = 10,
        linesearch = LineSearch.StrongWolfeLineSearch(), kwargs...
    ) where {Backend, LocalOpt <: Union{LBFGS, BFGS}}

    sol_pso = SciMLBase.solve!(cache.pso_cache; maxiters = maxiters)

    prob = remake(cache.prob, lb = nothing, ub = nothing)

    x0s = sol_pso.original
    grad_f = instantiate_gradient(prob.f.f, prob.f.adtype)

    result = cache.start_points
    copyto!(result, x0s)

    nlprob = SimpleNonlinearSolve.ImmutableNonlinearProblem{false}(grad_f, prob.u0, prob.p)
    nlalg = opt.local_opt isa LBFGS ?
        NonlinearSolveQuasiNewton.LimitedMemoryBroyden(;
            threshold = opt.local_opt.threshold,
            linesearch
        ) :
        NonlinearSolveQuasiNewton.Broyden(; linesearch)

    t0 = time()
    kernel = simplebfgs_run!(opt.backend)
    kernel(
        nlprob, x0s, result, nlalg, local_maxiters, abstol, reltol, grad_f;
        ndrange = length(x0s)
    )
    KernelAbstractions.synchronize(opt.backend)

    result_fx = (x -> prob.f(x, prob.p)).(result)
    result_fx = (x -> isnan(x) ? convert(eltype(prob.u0), Inf) : x).(result_fx)

    minobj, ind = findmin(result_fx)
    best_obj = sol_pso.objective isa Real ? sol_pso.objective : sol_pso.objective[]
    best_u, best_obj = minobj > best_obj ? (sol_pso.u, best_obj) : (result[ind], minobj)

    solve_time = (time() - t0) + sol_pso.stats.time
    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt, best_u, best_obj;
        stats = Optimization.OptimizationStats(; time = solve_time)
    )
end
