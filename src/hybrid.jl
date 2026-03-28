import Optim

@kernel function simplebfgs_run!(nlprob, x0s, result, opt, maxiters, abstol, reltol)
    i = @index(Global, Linear)
    nlcache = remake(nlprob; u0 = x0s[i])
    sol = solve(nlcache, opt; maxiters, abstol, reltol)
    @inbounds result[i] = sol.u
end

function SciMLBase.solve!(
        cache::HybridPSOCache, opt::HybridPSO{Backend, LocalOpt}, args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 100,
        local_maxiters = 50,
        n_starts = 20,
        kwargs...
    ) where {
        Backend, LocalOpt <: Union{LBFGS, BFGS},
    }
    # PSO exploration
    pso_cache = cache.pso_cache
    sol_pso = solve!(pso_cache)
    x0s = sol_pso.original
    prob = cache.prob

    best_u   = sol_pso.u
    best_obj = sol_pso.objective isa Real ? sol_pso.objective : sol_pso.objective[]

    # Rank starting points by objective value
    costs = map(x -> prob.f(x, prob.p), x0s)
    costs = map(c -> (isnan(c) || isinf(c)) ? convert(eltype(best_obj), Inf) : c, costs)
    n = min(n_starts, length(x0s))
    top_idx = partialsortperm(Vector(costs), 1:n)

    # Multi-start L-BFGS minimization 
    local_method = if opt.local_opt isa LBFGS
        Optim.LBFGS(; m = opt.local_opt.threshold)
    else
        Optim.BFGS()
    end

    _abstol = something(abstol, 1e-10)
    _reltol = something(reltol, 1e-10)
    orig_lb = prob.lb
    orig_ub = prob.ub

    optf = OptimizationFunction{false}(prob.f.f, prob.f.adtype)

    t0 = time()
    for i in top_idx
        u0 = x0s[i]
        # Nudge points on the boundary inward
        if orig_lb !== nothing
            ε = convert(eltype(u0), 1e-12)
            u0 = clamp.(u0, orig_lb .+ ε, orig_ub .- ε)
        end

        local_prob = if orig_lb !== nothing
            OptimizationProblem{false}(optf, u0, prob.p; lb = orig_lb, ub = orig_ub)
        else
            OptimizationProblem{false}(optf, u0, prob.p)
        end

        try
            sol = Optimization.solve(local_prob, local_method;
                maxiters = local_maxiters, abstol = _abstol, reltol = _reltol)
            fval = sol.objective isa Real ? sol.objective : sol.objective[]
            if !isnan(fval) && !isinf(fval) && fval < best_obj
                best_obj = fval
                best_u   = sol.u
            end
        catch
            continue
        end
    end
    t1 = time()

    solve_time = (t1 - t0) + sol_pso.stats.time
    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        best_u, best_obj,
        stats = Optimization.OptimizationStats(; time = solve_time)
    )
end