using KernelAbstractions
using SciMLBase
using Optimization
using LineSearch
using SimpleNonlinearSolve
using NonlinearSolveBase: ImmutableNonlinearProblem

@inline _unwrap_scalar(x::Real) = x
@inline _unwrap_scalar(x) = x[]

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

@kernel function simplebfgs_run!(
        grad_f, f, p, x0s, result, result_fx, nlalg, maxiters, abstol, reltol
    )
    i = @index(Global, Linear)
    @inbounds x0 = SVector(x0s[i])
    nlprob_i = ImmutableNonlinearProblem{false}(grad_f, x0, p)
    # `grad_f` kwarg required by LineSearch static Strong Wolfe cache (same as `prob.f`)
    sol = SciMLBase.solve(nlprob_i, nlalg; maxiters, abstol, reltol, grad_f)
    @inbounds x = as_svector(sol.u)
    T = eltype(x)
    @inbounds result[i] = x
    v = f(x, p)
    @inbounds result_fx[i] = (isnan(v) | !isfinite(v)) ? T(Inf) : convert(T, v)
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
    lb = prob.lb === nothing ? nothing : SVector{d, T}(prob.lb)
    ub = prob.ub === nothing ? nothing : SVector{d, T}(prob.ub)

    raw_grad = instantiate_gradient(f_raw, prob.f.adtype)
    grad_f = BoundedGrad(raw_grad, lb, ub)

    nlalg = if opt.local_opt isa LBFGS
        SimpleLimitedMemoryBroyden(;
            threshold = opt.local_opt.threshold,
            linesearch,
        )
    else
        SimpleBroyden(; linesearch)
    end

    x0s = sol_pso.original
    n = length(x0s)

    result = similar(x0s)
    result_fx = KernelAbstractions.allocate(opt.backend, T, n)

    t0 = time()

    kernel = simplebfgs_run!(opt.backend)
    kernel(
        grad_f, f_raw, p,
        x0s, result, result_fx,
        nlalg, local_maxiters, abstol, reltol;
        ndrange = n,
    )
    KernelAbstractions.synchronize(opt.backend)

    minobj, ind = findmin(result_fx)
    if minobj < best_obj
        best_obj = minobj
        best_u = if KernelAbstractions.isgpu(opt.backend)
            Array(result)[ind]
        else
            copy(result[ind])
        end
    end

    solve_time = (time() - t0) + sol_pso.stats.time
    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        best_u, best_obj;
        stats = Optimization.OptimizationStats(; time = solve_time),
    )
end
