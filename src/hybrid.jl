using LinearAlgebra: norm, dot
using KernelAbstractions
using SciMLBase
using Optimization
import LineSearch

# Isbits, GPU-safe eval functor for Strong Wolfe line search.
# Clamps to bounds, NaN firewall prevents non-finite values from poisoning interpolation.
struct OptEval{F, G, P, X, D, LB, UB}
    f::F
    grad_f::G
    p::P
    x::X
    dir::D
    lb::LB
    ub::UB
end

@inline function (e::OptEval)(α)
    xn = clamp.(e.x .+ α .* e.dir, e.lb, e.ub)
    v = e.f(xn, e.p)
    ϕ = ifelse(isfinite(v), v, eltype(e.x)(Inf))
    g = e.grad_f(xn, e.p)
    g = map(gi -> ifelse(isfinite(gi), gi, zero(gi)), g)
    dϕ = dot(g, e.dir)
    return (ϕ, dϕ)
end

# L-BFGS two-loop recursion (Nocedal & Wright Algorithm 7.4).
# NTuple history buffers live in registers on GPU.
@inline function _lbfgs_dir(g, S, Y, Rho, ::Val{M}, k) where {M}
    T = eltype(g)
    q, a = g, ntuple(_ -> zero(T), Val(M))
    for j in 0:(M - 1)
        idx = k - j
        if idx >= 1
            ii = mod1(idx, M)
            a = Base.setindex(a, Rho[ii] * dot(S[ii], q), ii)
            q = q - a[ii] * Y[ii]
        end
    end
    kk = mod1(k, M)
    sy, yy = dot(S[kk], Y[kk]), sum(abs2, Y[kk])
    γ = sy / yy
    γ = ifelse(k >= 1 && yy > T(1.0e-30) && isfinite(γ) && γ > zero(T), γ, one(T))
    r = γ * q
    for j in (M - 1):-1:0
        idx = k - j
        if idx >= 1
            ii = mod1(idx, M)
            r = r + (a[ii] - Rho[ii] * dot(Y[ii], r)) * S[ii]
        end
    end
    return -r
end

# Per-particle L-BFGS with Strong Wolfe line search.
# Each thread runs an independent local refinement from its own starting point.
@kernel function lbfgs_kernel!(f, grad_f, p, x0s, result, result_fx, lb, ub, maxiters, ::Val{M}) where {M}
    i = @index(Global, Linear)
    x = clamp.(x0s[i], lb, ub)
    T = eltype(x)
    z = zero(typeof(x))
    S, Y = ntuple(_ -> z, Val(M)), ntuple(_ -> z, Val(M))
    Rho = ntuple(_ -> zero(T), Val(M))
    fx = let v = f(clamp.(x, lb, ub), p); ifelse(isfinite(v), v, T(Inf)) end
    g = map(gi -> ifelse(isfinite(gi), gi, zero(gi)), grad_f(clamp.(x, lb, ub), p))
    k, active = 0, isfinite(fx) && all(isfinite, g)
    c1, c2 = T(1.0e-4), T(0.9)
    α_max = T(65536)
    ls_maxiters = 10
    zoom_maxiters = 10
    for _ in 1:maxiters
        if active && norm(g) >= T(1.0e-7)
            dir = _lbfgs_dir(g, S, Y, Rho, Val(M), k)
            if dot(g, dir) >= zero(T)
                dir, k = -g, 0
            end
            eval_fn = OptEval(f, grad_f, p, x, dir, lb, ub)
            ϕ_0, dϕ_0 = eval_fn(zero(T))
            α, ok = LineSearch._sw_search(
                eval_fn, ϕ_0, dϕ_0, c1, c2,
                one(T), α_max, ls_maxiters, zoom_maxiters
            )
            if !ok
                eval_fn = OptEval(f, grad_f, p, x, -g, lb, ub)
                ϕ_0, dϕ_0 = eval_fn(zero(T))
                α, ok = LineSearch._sw_search(
                    eval_fn, ϕ_0, dϕ_0, c1, c2,
                    one(T), α_max, ls_maxiters, zoom_maxiters
                )
                dir = -g
                k = 0
            end
            if ok
                xn = clamp.(x .+ α .* dir, lb, ub)
                fn = let v = f(xn, p); ifelse(isfinite(v), v, T(Inf)) end
                gn = map(gi -> ifelse(isfinite(gi), gi, zero(gi)), grad_f(xn, p))
            else
                xn, fn, gn = x, fx, g
            end
            if ok && isfinite(fn) && all(isfinite, gn)
                s, y = xn - x, gn - g
                sy = dot(s, y)
                if isfinite(one(T) / sy) && sy > T(1.0e-10)
                    k += 1; ii = mod1(k, M)
                    S = Base.setindex(S, s, ii)
                    Y = Base.setindex(Y, y, ii)
                    Rho = Base.setindex(Rho, one(T) / sy, ii)
                else
                    k = 0
                end
                x, g, fx = xn, gn, fn
            else
                active = false
            end
        end
    end
    @inbounds result[i] = x
    @inbounds result_fx[i] = fx
end

# HybridPSO: global PSO exploration, then per-particle L-BFGS+StrongWolfe refinement on GPU.
function SciMLBase.solve!(
        cache::HybridPSOCache, opt::HybridPSO{Backend, LocalOpt}, args...;
        abstol = nothing, reltol = nothing, maxiters = 100,
        local_maxiters = 50, kwargs...
    ) where {Backend, LocalOpt <: Union{LBFGS, BFGS}}

    sol_pso = SciMLBase.solve!(cache.pso_cache; maxiters = maxiters)

    prob = cache.prob
    f_raw, p = prob.f.f, prob.p
    lb = prob.lb === nothing ? convert.(eltype(prob.u0), -Inf) : prob.lb
    ub = prob.ub === nothing ? convert.(eltype(prob.u0), Inf) : prob.ub

    best_u = sol_pso.u
    best_obj = sol_pso.objective isa Real ? sol_pso.objective : sol_pso.objective[]

    x0s = sol_pso.original
    n = length(x0s)
    m_val = length(prob.u0) > 20 ? Val(5) : Val(10)

    grad_f = instantiate_gradient(f_raw, prob.f.adtype)
    t0 = time()

    result = similar(x0s)
    result_fx = KernelAbstractions.allocate(opt.backend, typeof(best_obj), n)

    kernel = lbfgs_kernel!(opt.backend)
    kernel(f_raw, grad_f, p, x0s, result, result_fx, lb, ub, local_maxiters, m_val; ndrange = n)
    KernelAbstractions.synchronize(opt.backend)

    minobj, ind = findmin(result_fx)
    if minobj < best_obj
        best_obj = minobj
        best_u = Array(view(result, ind:ind))[1]
    end

    solve_time = (time() - t0) + sol_pso.stats.time
    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt, best_u, best_obj;
        stats = Optimization.OptimizationStats(; time = solve_time)
    )
end