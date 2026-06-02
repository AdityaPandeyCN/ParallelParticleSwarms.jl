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

struct BoundedGrad{G, LB, UB}
    raw::G
    lb::LB
    ub::UB
end

@inline function (bg::BoundedGrad)(θ, p)
    T = eltype(θ)
    in_box = all(isfinite, θ) &&
             all(θ .>= bg.lb .- T(2) .* (bg.ub .- bg.lb)) &&
             all(θ .<= bg.ub .+ T(2) .* (bg.ub .- bg.lb))
    g = in_box ? bg.raw(θ, p) : map(_ -> T(1.0e15), θ)
    return as_svector(g)
end

@inline function _nlalg(local_opt::LBFGS, linesearch)
    SimpleLimitedMemoryBroyden(; threshold = local_opt.threshold, linesearch)
end
@inline _nlalg(::BFGS, linesearch) = SimpleBroyden(; linesearch)

@inline function _nlprob(grad_f, u0, p)
    convert(
        ImmutableNonlinearProblem,
        SciMLBase.NonlinearProblem{false}(grad_f, as_svector(u0), p),
    )
end

@inline function _local_solve(grad_f, u0, p, nlalg, maxiters, abstol, reltol, grad_f_kw)
    nlprob = _nlprob(grad_f, u0, p)
    if grad_f_kw
        return solve(nlprob, nlalg; maxiters, abstol, reltol, grad_f)
    end
    return solve(nlprob, nlalg; maxiters, abstol, reltol)
end

@kernel function simplebfgs_run!(
        grad_f, f_raw, p, x0s, result, result_fx, nlalg,
        maxiters, abstol, reltol, grad_f_kw::Bool,
    )
    i = @index(Global, Linear)
    sol = _local_solve(grad_f, x0s[i], p, nlalg, maxiters, abstol, reltol, grad_f_kw)
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
    lb, ub = _hybrid_bounds(prob, Val(length(prob.u0)), T)
    backend = opt.backend

    grad_f = as_svector_grad(BoundedGrad(instantiate_gradient(f_raw, prob.f.adtype), lb, ub))
    nlalg = _nlalg(opt.local_opt, linesearch)
    grad_f_kw = linesearch isa StrongWolfeLineSearch

    x0s = sol_pso.original
    n = length(x0s)
    result = cache.start_points
    copyto!(result, x0s)
    result_fx = KernelAbstractions.allocate(backend, T, n)

    t0 = time()
    simplebfgs_run!(backend)(
        grad_f, f_raw, p, x0s, result, result_fx, nlalg,
        local_maxiters, abstol, reltol, grad_f_kw;
        ndrange = n,
    )
    KernelAbstractions.synchronize(backend)

    fx_host = Array(result_fx)
    minobj, ind = findmin(fx_host)
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
