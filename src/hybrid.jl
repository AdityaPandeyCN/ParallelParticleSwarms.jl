using LinearAlgebra: dot
using KernelAbstractions
using SciMLBase
using Optimization
using LineSearch

"Check that `θ` lies within a slack-expanded box around `[lb, ub]`."
@inline _in_safe_box(θ, ::Nothing, ::Nothing) = all(isfinite, θ)
@inline function _in_safe_box(θ::AbstractArray{T}, lb, ub) where {T}
    all(isfinite, θ) || return false
    w = ub .- lb
    return all(θ .>= lb .- T(2) .* w) && all(θ .<= ub .+ T(2) .* w)
end

"Sentinel gradient with huge magnitude used to reject out-of-box trial points."
@inline _huge_grad(θ::AbstractArray{T}) where {T} = map(_ -> T(1.0e15), θ)

"L-BFGS two-loop recursion (Nocedal & Wright Algorithm 7.4) over cyclic memory of length `M`."
@inline function _lbfgs_direction(g, S, Y, Rho, ::Val{M}, k) where {M}
    T = eltype(g)
    q = g
    a = ntuple(_ -> zero(T), Val(M))
    for j in 0:(M - 1)
        idx = k - j
        if idx >= 1
            ii = mod1(idx, M)
            aii = Rho[ii] * dot(S[ii], q)
            a = Base.setindex(a, aii, ii)
            q = q - aii * Y[ii]
        end
    end
    γ = if k >= 1
        kk = mod1(k, M)
        yy = sum(abs2, Y[kk])
        sy = dot(S[kk], Y[kk])
        ifelse(yy > T(1.0e-30) && sy > zero(T), sy / yy, one(T))
    else
        one(T)
    end
    r = γ * q
    for j in (M - 1):-1:0
        idx = k - j
        if idx >= 1
            ii = mod1(idx, M)
            β = Rho[ii] * dot(Y[ii], r)
            r = r + (a[ii] - β) * S[ii]
        end
    end
    return -r
end

"Per-particle hand-written L-BFGS with `LineSearch.StrongWolfeLineSearch`. Avoids `SciMLBase.init`, whose `promote_u0` path uses dynamic dispatch (invalid GPU IR)."
@kernel function lbfgs_run!(
        grad_f, p, x0s, result, ls_cache, ::Val{M}, maxiters
    ) where {M}
    i = @index(Global, Linear)
    x = x0s[i]
    T = eltype(x)
    g = grad_f(x, p)

    z = zero(typeof(x))
    S = ntuple(_ -> z, Val(M))
    Y = ntuple(_ -> z, Val(M))
    Rho = ntuple(_ -> zero(T), Val(M))
    k = 0
    active = all(isfinite, g)

    for _ in 1:maxiters
        if active
            dir = _lbfgs_direction(g, S, Y, Rho, Val(M), k)
            if dot(g, dir) >= zero(T)
                dir = -g
                k = 0
            end
            ls_sol = SciMLBase.solve!(ls_cache, x, dir)
            α = T(ls_sol.step_size)
            ok = ls_sol.retcode == ReturnCode.Success
            if ok && isfinite(α) && α > zero(T)
                xn = x + α * dir
                gn = grad_f(xn, p)
                if all(isfinite, gn)
                    s = xn - x
                    y = gn - g
                    sy = dot(s, y)
                    if sy > T(1.0e-10) && isfinite(one(T) / sy)
                        k += 1
                        ii = mod1(k, M)
                        S = Base.setindex(S, s, ii)
                        Y = Base.setindex(Y, y, ii)
                        Rho = Base.setindex(Rho, one(T) / sy, ii)
                    else
                        k = 0
                    end
                    x = xn
                    g = gn
                else
                    active = false
                end
            else
                active = false
            end
        end
    end
    @inbounds result[i] = x
end

"Hybrid PSO solve: global PSO exploration followed by per-particle local L-BFGS refinement."
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

    T = eltype(prob.u0)
    D = length(prob.u0)
    M_val = opt.local_opt isa LBFGS ? Val(min(opt.local_opt.threshold, D)) : Val(D)

    # Construct the line-search cache directly: `LineSearch.init` would route
    # through `SciMLBase.init`'s `promote_u0`, which dynamic-dispatches on GPU.
    ls_cache = LineSearch.StaticStrongWolfeLineSearchCache(
        grad_f, grad_f, prob.p,
        T(linesearch.c1), T(linesearch.c2),
        T(linesearch.α_init), T(linesearch.α_max),
        linesearch.maxiters, linesearch.zoom_maxiters
    )

    t0 = time()
    kernel = lbfgs_run!(opt.backend)
    kernel(
        grad_f, prob.p, x0s, result, ls_cache, M_val, local_maxiters;
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
