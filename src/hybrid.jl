using KernelAbstractions
using SciMLBase
using Optimization
using LineSearch
using NonlinearSolveQuasiNewton

# Safety box around the original feasible region. Local refinement may step
# modestly outside `[lb, ub]`, but past this margin we reject the trial point
# instead of evaluating f there — e.g. F19 Griewank-Rosenbrock overflows to Inf
# for |θ| ~ 1e10 in Float32 and then cos(Inf) throws DomainError.
@inline _in_safe_box(θ, ::Nothing, ::Nothing) = all(isfinite, θ)
@inline function _in_safe_box(θ::AbstractArray{T}, lb, ub) where {T}
    all(isfinite, θ) || return false
    w = ub .- lb
    return all(θ .>= lb .- T(2) .* w) && all(θ .<= ub .+ T(2) .* w)
end

# Huge-magnitude gradient: makes the Strong Wolfe Armijo test fail so the line
# search rejects the trial step without us ever calling f at the bad point.
@inline _huge_grad(θ::AbstractArray{T}) where {T} = map(_ -> T(1.0e15), θ)

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

    orig_lb = cache.prob.lb
    orig_ub = cache.prob.ub
    prob = remake(cache.prob, lb = nothing, ub = nothing)

    x0s = sol_pso.original
    raw_grad = instantiate_gradient(prob.f.f, prob.f.adtype)
    grad_f = (θ, p) ->
        _in_safe_box(θ, orig_lb, orig_ub) ? raw_grad(θ, p) : _huge_grad(θ)

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

    Tobj = eltype(prob.u0)
    result_fx = map(eachindex(result)) do i
        x = result[i]
        _in_safe_box(x, orig_lb, orig_ub) || return convert(Tobj, Inf)
        v = prob.f(x, prob.p)
        isnan(v) ? convert(Tobj, Inf) : v
    end

    minobj, ind = findmin(result_fx)
    best_obj = sol_pso.objective isa Real ? sol_pso.objective : sol_pso.objective[]
    best_u, best_obj = minobj > best_obj ? (sol_pso.u, best_obj) : (result[ind], minobj)

    solve_time = (time() - t0) + sol_pso.stats.time
    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt, best_u, best_obj;
        stats = Optimization.OptimizationStats(; time = solve_time)
    )
end
