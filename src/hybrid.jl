using KernelAbstractions
using SciMLBase
using Optimization
using LineSearch
using NonlinearSolveQuasiNewton

"Check that `θ` lies within a slack-expanded box around `[lb, ub]`."
@inline _in_safe_box(θ, ::Nothing, ::Nothing) = all(isfinite, θ)
@inline function _in_safe_box(θ::AbstractArray{T}, lb, ub) where {T}
    all(isfinite, θ) || return false
    w = ub .- lb
    return all(θ .>= lb .- T(2) .* w) && all(θ .<= ub .+ T(2) .* w)
end

"Sentinel gradient with huge magnitude used to reject out-of-box trial points."
@inline _huge_grad(θ::AbstractArray{T}) where {T} = map(_ -> T(1.0e15), θ)

"Per-particle local quasi-Newton refinement kernel. The algorithm is built inside the kernel because `QuasiNewtonAlgorithm` is not isbits."
@kernel function simplebfgs_run!(
        nlprob, x0s, result, linesearch, ::Val{Threshold}, ::Val{IsLBFGS},
        maxiters, abstol, reltol, grad_f
    ) where {Threshold, IsLBFGS}
    i = @index(Global, Linear)
    nlalg = IsLBFGS ?
        NonlinearSolveQuasiNewton.LimitedMemoryBroyden(; threshold = Val(Threshold), linesearch) :
        NonlinearSolveQuasiNewton.Broyden(; linesearch)
    nlcache = SciMLBase.init(
        nlprob, nlalg; u0 = x0s[i], maxiters, abstol, reltol, grad_f,
        kwargshandle = SciMLBase.KeywordArgSilent
    )
    sol = SciMLBase.solve!(nlcache)
    @inbounds result[i] = sol.u
end

"Hybrid PSO solve: global PSO exploration followed by per-particle local quasi-Newton refinement."
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
    is_lbfgs_val = Val(opt.local_opt isa LBFGS)
    threshold_val = opt.local_opt isa LBFGS ? Val(opt.local_opt.threshold) : Val(0)

    t0 = time()
    kernel = simplebfgs_run!(opt.backend)
    kernel(
        nlprob, x0s, result, linesearch, threshold_val, is_lbfgs_val,
        local_maxiters, abstol, reltol, grad_f;
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
