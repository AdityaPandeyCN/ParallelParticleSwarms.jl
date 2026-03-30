using LinearAlgebra: norm, dot

@inline function _safe_eval(f, p, x, lb, ub)
    T = eltype(x)
    eps = T(1e-12)
    xc = clamp.(x, lb .+ eps, ub .- eps)
    xc = map(xi -> abs(xi) < eps ? eps : xi, xc)
    v = f(xc, p)
    isfinite(v) ? v : T(Inf)
end

@inline function _safe_grad(grad_f, p, x, lb, ub)
    T = eltype(x)
    eps = T(1e-12)
    xc = clamp.(x, lb .+ eps, ub .- eps)
    xc = map(xi -> abs(xi) < eps ? eps : xi, xc)
    g = grad_f(xc, p)
    map(gi -> isfinite(gi) ? gi : zero(gi), g)
end

@inline function _backtrack(f, grad_f, p, x, fx, g, dir, lb, ub)
    T = eltype(x)
    slope = dot(g, dir)
    slope >= zero(T) && return x, fx, g, false
    a = one(T)
    for _ in 1:30
        xn = clamp.(x + a * dir, lb, ub)
        fn = _safe_eval(f, p, xn, lb, ub)
        fn <= fx + T(1e-4) * a * slope && return xn, fn, _safe_grad(grad_f, p, xn, lb, ub), true
        a *= T(0.5)
    end
    return x, fx, g, false
end

@inline function _lbfgs_dir(g, S, Y, Rho, ::Val{M}, k) where {M}
    T = eltype(g)
    q = g
    a = ntuple(_ -> zero(T), Val(M))
    for j in 0:M-1
        idx = k - j; idx < 1 && continue
        ii = mod1(idx, M)
        a  = Base.setindex(a, Rho[ii] * dot(S[ii], q), ii)
        q  = q - a[ii] * Y[ii]
    end
    kk = mod1(k, M)
    yy = sum(abs2, Y[kk])
    gamma = (k >= 1 && yy > T(1e-30)) ? dot(S[kk], Y[kk]) / yy : one(T)
    gamma = (isfinite(gamma) && gamma > zero(T)) ? gamma : one(T)
    r = gamma * q
    for j in M-1:-1:0
        idx = k - j; idx < 1 && continue
        ii = mod1(idx, M)
        r  = r + (a[ii] - Rho[ii] * dot(Y[ii], r)) * S[ii]
    end
    return -r
end

@kernel function lbfgs_kernel!(f, grad_f, p, x0s, result, lb, ub, maxiters, ::Val{M}) where {M}
    i = @index(Global, Linear)
    x = clamp.(x0s[i], lb, ub)
    T = eltype(x)
    z = zero(typeof(x))
    S, Y = ntuple(_ -> z, Val(M)), ntuple(_ -> z, Val(M))
    Rho  = ntuple(_ -> zero(T), Val(M))
    fx = _safe_eval(f, p, x, lb, ub)
    if isfinite(fx)
        g = _safe_grad(grad_f, p, x, lb, ub)
        if all(isfinite, g)
            k = 0
            for _ in 1:maxiters
                norm(g) < T(1e-10) && break
                dir = _lbfgs_dir(g, S, Y, Rho, Val(M), k)
                dot(g, dir) >= zero(T) && (dir = -g; k = 0)
                xn, fn, gn, ok = _backtrack(f, grad_f, p, x, fx, g, dir, lb, ub)
                if !ok
                    xn, fn, gn, ok = _backtrack(f, grad_f, p, x, fx, g, -g, lb, ub)
                    k = 0; !ok && break
                end
                (!isfinite(fn) || any(!isfinite, gn)) && break
                s, y = xn - x, gn - g
                sy = dot(s, y)
                if isfinite(one(T) / sy) && sy > T(1e-10)
                    k += 1; ii = mod1(k, M)
                    S   = Base.setindex(S, s, ii)
                    Y   = Base.setindex(Y, y, ii)
                    Rho = Base.setindex(Rho, one(T) / sy, ii)
                else
                    k = 0
                end
                x, g, fx = xn, gn, fn
            end
        end
    end
    @inbounds result[i] = x
end

function SciMLBase.solve!(
        cache::HybridPSOCache, opt::HybridPSO{Backend, LocalOpt}, args...;
        abstol = nothing, reltol = nothing, maxiters = 100,
        local_maxiters = 50, n_starts = 20, kwargs...
    ) where {Backend, LocalOpt <: Union{LBFGS, BFGS}}

    sol_pso  = solve!(cache.pso_cache)
    prob     = cache.prob
    f_raw, p = prob.f.f, prob.p
    lb, ub   = prob.lb, prob.ub
    best_u   = sol_pso.u
    best_obj = sol_pso.objective isa Real ? sol_pso.objective : sol_pso.objective[]

    _obj(x) = let v = prob.f(clamp.(x, lb, ub), p)
        (isnan(v) || isinf(v)) ? convert(eltype(best_obj), Inf) : v
    end

    x0s   = sol_pso.original
    costs = map(_obj, x0s)
    n     = min(n_starts, length(x0s))
    pool  = [x0s[j] for j in partialsortperm(Vector(costs), 1:n)]
    D     = length(first(pool))
    m_val = D > 20 ? Val(5) : Val(10)

    grad_f = instantiate_gradient(f_raw, prob.f.adtype)
    t0 = time()

    result = similar(pool)
    copyto!(result, pool)
    kernel = lbfgs_kernel!(opt.backend)
    kernel(f_raw, grad_f, p, pool, result, lb, ub, local_maxiters, m_val; ndrange = n)
    KernelAbstractions.synchronize(opt.backend)

    for j in 1:n
        r    = clamp.(result[j], lb, ub)
        fval = _obj(r)
        fval < best_obj && (best_obj = fval; best_u = r)
    end

    SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt, best_u, best_obj;
        stats = Optimization.OptimizationStats(; time = (time() - t0) + sol_pso.stats.time))
end