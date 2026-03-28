import Optim

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
    # Phase 1: PSO exploration
    pso_cache = cache.pso_cache
    sol_pso = solve!(pso_cache)
    x0s = sol_pso.original
    prob = cache.prob

    best_u   = sol_pso.u
    best_obj = sol_pso.objective isa Real ? sol_pso.objective : sol_pso.objective[]

    # Phase 2: rank starting points by objective value
    costs = map(x -> prob.f(x, prob.p), x0s)
    costs = map(c -> (isnan(c) || isinf(c)) ? convert(eltype(best_obj), Inf) : c, costs)
    n = min(n_starts, length(x0s))
    top_idx = partialsortperm(Vector(costs), 1:n)

    # Phase 3: multi-start L-BFGS minimization from top particles
    local_method = if opt.local_opt isa LBFGS
        Optim.LBFGS(; m = opt.local_opt.threshold)
    else
        Optim.BFGS()
    end

    _abstol = something(abstol, 1e-10)
    _reltol = something(reltol, 1e-10)
    orig_lb = prob.lb
    orig_ub = prob.ub

    # convert bounds to Vector for Optim.Fminbox compatibility
    _lb = orig_lb !== nothing ? Vector(orig_lb) : nothing
    _ub = orig_ub !== nothing ? Vector(orig_ub) : nothing

    # wrap objective to accept plain Vector
    _f = (u, p) -> prob.f.f(u, p)
    optf = OptimizationFunction(_f, AutoForwardDiff())

    t0 = time()
    for i in top_idx
        u0 = Vector(x0s[i])
        # nudge points on the boundary inward
        if _lb !== nothing
            ε = 1e-12
            u0 .= clamp.(u0, _lb .+ ε, _ub .- ε)
        end

        local_prob = if _lb !== nothing
            OptimizationProblem(optf, u0, prob.p; lb = _lb, ub = _ub)
        else
            OptimizationProblem(optf, u0, prob.p)
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