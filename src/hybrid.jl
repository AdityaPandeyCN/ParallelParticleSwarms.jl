using KernelAbstractions
using SciMLBase
using Optimization
using LineSearch
using SimpleNonlinearSolve
using NonlinearSolveBase: ImmutableNonlinearProblem

@inline _unwrap_scalar(x::Real) = x
@inline _unwrap_scalar(x) = x[]

function _hybrid_bounds(prob, ::Val{d}, T) where {d}
    lb = prob.lb === nothing ? SVector{d, T}(fill(-T(Inf), d)) : SVector{d, T}(prob.lb)
    ub = prob.ub === nothing ? SVector{d, T}(fill(T(Inf), d)) : SVector{d, T}(prob.ub)
    return lb, ub
end

"""Gradient wrapper: in-box values use AD; out-of-box trials get a huge sentinel gradient."""
struct BoundedGrad{G, LB, UB}
    raw::G
    lb::LB
    ub::UB
end

@inline function (bg::BoundedGrad{G, Nothing, Nothing})(θ, p) where {G}
    return as_svector(bg.raw(θ, p))
end

@inline function (bg::BoundedGrad)(θ, p)
    T = eltype(θ)
    in_box = all(isfinite, θ) &&
             all(θ .>= bg.lb .- T(2) .* (bg.ub .- bg.lb)) &&
             all(θ .<= bg.ub .+ T(2) .* (bg.ub .- bg.lb))
    g = in_box ? bg.raw(θ, p) : map(_ -> T(1.0e15), θ)
    return as_svector(g)
end

@inline function _local_nlalg(local_opt::LBFGS, linesearch)
    return SimpleLimitedMemoryBroyden(; threshold = local_opt.threshold, linesearch)
end
@inline _local_nlalg(::BFGS, linesearch) = SimpleBroyden(; linesearch)

# Per-particle local solve via SimpleNonlinearSolve (same API as standalone `LBFGS`/`BFGS`).
@kernel function simplebfgs_run!(
        grad_f, f_raw, p, x0s, result, result_fx, nlalg, maxiters, abstol, reltol
    )
    i = @index(Global, Linear)
    x0 = as_svector(x0s[i])
    nlprob_i = ImmutableNonlinearProblem{false}(grad_f, x0, p)
    sol = SciMLBase.solve(
        nlprob_i,
        nlalg;
        maxiters,
        abstol,
        reltol,
        grad_f,
        kwargshandle = SciMLBase.KeywordArgSilent,
    )
    u = as_svector(sol.u)
    T = eltype(u)
    v = f_raw(u, p)
    fx = (isnan(v) | !isfinite(v)) ? T(Inf) : convert(T, v)
    @inbounds result[i] = u
    @inbounds result_fx[i] = fx
end

function SciMLBase.solve!(
        cache::HybridPSOCache, opt::HybridPSO{Backend, LocalOpt}, args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 100,
        local_maxiters = 50,
        linesearch = StrongWolfeLineSearch(),
        kwargs...
    ) where {Backend, LocalOpt <: Union{LBFGS, BFGS}}

    sol_pso = SciMLBase.solve!(cache.pso_cache; maxiters)
    best_u = sol_pso.u
    best_obj = _unwrap_scalar(sol_pso.objective)

    prob = cache.prob
    f_raw, p = prob.f.f, prob.p
    T = eltype(prob.u0)
    d = length(prob.u0)
    lb, ub = _hybrid_bounds(prob, Val(d), T)

    raw_grad = instantiate_gradient(f_raw, prob.f.adtype)
    grad_f = BoundedGrad(raw_grad, lb, ub)
    nlalg = _local_nlalg(opt.local_opt, linesearch)

    x0s = sol_pso.original
    n = length(x0s)
    result = cache.start_points
    copyto!(result, x0s)
    result_fx = KernelAbstractions.allocate(opt.backend, T, n)

    t0 = time()
    kernel = simplebfgs_run!(opt.backend)
    kernel(
        grad_f, f_raw, p, x0s, result, result_fx, nlalg,
        local_maxiters, abstol, reltol;
        ndrange = n,
    )
    KernelAbstractions.synchronize(opt.backend)

    minobj, ind = findmin(Array(result_fx))
    if minobj < best_obj
        best_obj = minobj
        best_u = Array(result)[ind]
    end

    solve_time = (time() - t0) + sol_pso.stats.time
    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        best_u, best_obj;
        stats = Optimization.OptimizationStats(; time = solve_time),
    )
end
