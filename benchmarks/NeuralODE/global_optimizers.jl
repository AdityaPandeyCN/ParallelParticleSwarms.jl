using Pkg
Pkg.activate(@__DIR__)

using SimpleChains
using StaticArrays
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using OptimizationSciPy
using ParallelParticleSwarms
using CUDA
using KernelAbstractions
using Adapt
using Random

# Optional: set CUDA device like the other scripts do
if haskey(ENV, "CUDA_DEVICE")
    CUDA.device!(parse(Int, ENV["CUDA_DEVICE"]))
end

CUDA.allowscalar(false)

# ---- Dataset: Spiral ODE (paper setup) ----
u0 = @SArray Float32[2.0, 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODE(u, p, t)
    true_A = @SMatrix Float32[-0.1 2.0; -2.0 -0.1]
    ((u .^ 3)'true_A)'
end

data_prob = ODEProblem(trueODE, u0, tspan)
data = Array(solve(data_prob, Tsit5(), saveat = tsteps))

# ---- Model: SimpleChains neural ODE ----
sc = SimpleChain(static(2),
    Activation(x -> x .^ 3),
    TurboDense{true}(tanh, static(2)),
    TurboDense{true}(identity, static(2)))

rng = Random.default_rng()
Random.seed!(rng, 0)
p_nn = SimpleChains.init_params(sc; rng)

f(u, p, t) = sc(u, p)
sprob_nn = ODEProblem(f, u0, tspan)

function predict_neuralode(p)
    Array(solve(sprob_nn,
        Tsit5();
        p = p,
        saveat = tsteps,
        sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP())))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, data .- pred)
    return loss, pred
end

# Optimization.jl expects a scalar objective for gradients.
loss_only(p) = first(loss_neuralode(p))

# ---- Gradient-based baselines (existing style) ----
optf = Optimization.OptimizationFunction((x, p) -> loss_only(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, p_nn)

@info "Adam (100 iters)"
@time res_adam = Optimization.solve(optprob, ADAM(0.05), maxiters = 100)
@show res_adam.objective
@show res_adam.stats.time

@info "LBFGS (100 iters)"
moptprob = OptimizationProblem(optf, MArray{Tuple{size(p_nn)...}}(p_nn...))
@time res_lbfgs = Optimization.solve(moptprob, LBFGS(), maxiters = 100)
@show res_lbfgs.objective
@show res_lbfgs.stats.time

# ---- SciPy global optimizers (requested) ----
# We use a Float64 vector for SciPy interop, but evaluate loss in Float32 internally.
p_static = SArray{Tuple{size(p_nn)...}}(p_nn...)

function nn_fn(u::T, p, t)::T where {T}
    nn, ps = p
    return nn(u, ps)
end

prob_nn = ODEProblem(nn_fn, u0, tspan, (sc, p_static))

function loss_scipy(u, p)
    odeprob, t = p
    u32 = Float32.(u)
    prob = remake(odeprob; p = (odeprob.p[1], u32))
    pred = Array(solve(prob, Tsit5(), saveat = t))
    sum(abs2, data .- pred)
end

# Finite bounds are important for SciPy global optimizers
lb = fill(-10.0, length(p_static))
ub = fill(10.0, length(p_static))
x0 = Float64.(collect(p_static))

scipy_prob = OptimizationProblem(loss_scipy, x0, (prob_nn, collect(Float32.(tsteps))); lb = lb, ub = ub)

@info "SciPy DE (maxiters=200)"
@time sol_de = Optimization.solve(scipy_prob, Scipy_DE(), maxiters = 200)
@show sol_de.objective
@show sol_de.stats.time

@info "SciPy DualAnnealing (maxiters=200)"
@time sol_da = Optimization.solve(scipy_prob, Scipy_DualAnnealing(), maxiters = 200)
@show sol_da.objective
@show sol_da.stats.time

@info "SciPy SHGO (maxiters=200)"
@time sol_shgo = Optimization.solve(scipy_prob, Scipy_SHGO(), maxiters = 200)
@show sol_shgo.objective
@show sol_shgo.stats.time

# ---- GPU PSO (existing codepath via parameter_estim_ode!) ----
@info "GPU PSO parameter estimation (n_particles=10_000, maxiters=100)"
n_particles = 10_000
backend = CUDABackend()

function prob_func(prob, gpu_particle)
    remake(prob, p = (prob.p[1], gpu_particle.position))
end

psolb = @SArray fill(-10.0f0, length(p_static))
psoub = @SArray fill(10.0f0, length(p_static))
soptprob = OptimizationProblem(loss_scipy, p_static, (prob_nn, collect(Float32.(tsteps))); lb = psolb, ub = psoub)

Random.seed!(rng, 0)
opt = ParallelPSOKernel(n_particles)
gbest, particles = ParallelParticleSwarms.init_particles(soptprob, opt, typeof(u0))

gpu_data = adapt(backend,
    [SVector{length(u0), eltype(u0)}(@view data[:, i]) for i in 1:length(tsteps)])
gpu_particles = adapt(backend, particles)
losses = adapt(backend, ones(eltype(u0), (1, n_particles)))
solver_cache = (; losses, gpu_particles, gpu_data, gbest)

adaptive = true
@time gsol = ParallelParticleSwarms.parameter_estim_ode!(prob_nn,
    solver_cache,
    psolb,
    psoub, Val(adaptive);
    saveat = tsteps,
    dt = 0.1f0,
    prob_func = prob_func,
    maxiters = 100)

@show gsol.cost

