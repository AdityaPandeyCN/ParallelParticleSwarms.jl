using Pkg
Pkg.activate(@__DIR__)

using SimpleChains,
    StaticArrays, OrdinaryDiffEq, SciMLSensitivity, Optimization
using OptimizationOptimisers
using Optimisers: Adam
using OptimizationOptimJL
using OptimizationSciPy
using SciMLBase: ImmutableODEProblem

using ParallelParticleSwarms
using DiffEqGPU
using CUDA
using KernelAbstractions
using Adapt
using Random
using Setfield

device!(0)

println("=" ^ 60)
println("NEURAL ODE BENCHMARK - Global Optimization Comparison")
println("=" ^ 60)

u0 = @SArray Float32[2.0, 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODE(u, p, t)
    true_A = @SMatrix Float32[-0.1 2.0; -2.0 -0.1]
    return ((u .^ 3)'true_A)'
end

prob = ODEProblem(trueODE, u0, tspan)
data = Array(solve(prob, Tsit5(), saveat = tsteps))

sc = SimpleChain(
    static(2),
    Activation(x -> x .^ 3),
    TurboDense{true}(tanh, static(2)),
    TurboDense{true}(identity, static(2))
)

rng = Random.default_rng()
Random.seed!(rng, 0)
p_nn = SimpleChains.init_params(sc; rng)

f(u, p, t) = sc(u, p)
sprob_nn = ODEProblem(f, u0, tspan)

function predict_neuralode(p)
    return Array(
        solve(
            sprob_nn, Tsit5(); p = p, saveat = tsteps,
            sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP())
        )
    )
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, data .- pred)
    return loss, pred
end

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x)[1], Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, p_nn)

@info "Adam (100 iters)"
@time res_adam = Optimization.solve(optprob, Adam(0.05), maxiters = 100)
@show res_adam.objective

@info "LBFGS (100 iters)"
moptprob = OptimizationProblem(optf, MArray{Tuple{size(p_nn)...}}(p_nn...))
@time res_lbfgs = Optimization.solve(moptprob, LBFGS(), maxiters = 100)
@show res_lbfgs.objective

## SciPy Global Optimizers

function loss_scipy(u, _)
    u32 = Float32.(u)
    pred = Array(solve(sprob_nn, Tsit5(); p = u32, saveat = tsteps))
    return sum(abs2, data .- pred)
end

lb_scipy = fill(-10.0, length(p_nn))
ub_scipy = fill(10.0, length(p_nn))
x0_scipy = Float64.(p_nn)
scipy_prob = OptimizationProblem(loss_scipy, x0_scipy; lb = lb_scipy, ub = ub_scipy)

@info "SciPy Differential Evolution (maxiters=200)"
@time sol_de = Optimization.solve(scipy_prob, OptimizationSciPy.ScipyDifferentialEvolution(), maxiters = 200)
@show sol_de.objective

@info "SciPy Dual Annealing (maxiters=200)"
@time sol_da = Optimization.solve(scipy_prob, OptimizationSciPy.ScipyDualAnnealing(), maxiters = 200)
@show sol_da.objective

## GPU-PSO with ImmutableODEProblem

function nn_fn(u::T, p, t)::T where {T}
    nn, ps = p
    return nn(u, ps)
end

p_static = SArray{Tuple{size(p_nn)...}}(p_nn...)

# Create base problem - this is ODEProblem{false}
prob_nn = ODEProblem{false}(nn_fn, u0, tspan, (sc, p_static))

# Convert to ImmutableODEProblem (isbits compatible) - used for probs array only
improb = DiffEqGPU.make_prob_compatible(prob_nn)

n_particles = 10_000
backend = CUDABackend()

function loss_pso(u, p)
    return eltype(u)(Inf)  # Start with Inf so any real loss improves
end

lb = @SArray fill(Float32(-10.0), length(p_static))
ub = @SArray fill(Float32(10.0), length(p_static))

soptprob = OptimizationProblem(loss_pso, p_static, nothing; lb = lb, ub = ub)

Random.seed!(rng, 0)
opt = ParallelPSOKernel(n_particles)
gbest, particles = ParallelParticleSwarms.init_particles(soptprob, opt, typeof(p_static))

gpu_data = adapt(
    backend,
    [
        SVector{length(u0), eltype(u0)}(@view data[:, i])
        for i in 1:length(tsteps)
    ]
)

CUDA.allowscalar(false)

# prob_func that works with ImmutableODEProblem on GPU
function prob_func(prob, gpu_particle)
    return remake(prob, p = (prob.p[1], gpu_particle.position))
end

gpu_particles = adapt(backend, particles)
losses = adapt(backend, ones(eltype(prob.u0), n_particles))

# Pre-allocate probs as CuArray of ImmutableODEProblem
probs = adapt(backend, fill(improb, n_particles))

# Cache: 5 elements (probs is CuArray of ImmutableODEProblem)
solver_cache = (; losses, gpu_particles, gpu_data, gbest, probs)

@info "GPU-PSO Warmup (compilation)"
@time gsol = ParallelParticleSwarms.parameter_estim_ode!(
    prob_nn,  # Pass ODEProblem{false} - vectorized_asolve needs this as 2nd arg
    solver_cache, lb, ub, Val(true);
    saveat = tsteps, dt = 0.1f0, maxiters = 10,
    prob_func = prob_func
)

# Reset for fair benchmark
Random.seed!(rng, 0)
gbest, particles = ParallelParticleSwarms.init_particles(soptprob, opt, typeof(p_static))
gpu_particles = adapt(backend, particles)
losses = adapt(backend, ones(eltype(prob.u0), n_particles))
probs = adapt(backend, fill(improb, n_particles))
solver_cache = (; losses, gpu_particles, gpu_data, gbest, probs)

@info "GPU-PSO (n_particles=$n_particles, maxiters=100)"
@time gsol = ParallelParticleSwarms.parameter_estim_ode!(
    prob_nn,  # Pass ODEProblem{false} - vectorized_asolve needs this as 2nd arg
    solver_cache, lb, ub, Val(true);
    saveat = tsteps, dt = 0.1f0, maxiters = 100,
    prob_func = prob_func
)

@show gsol.cost

println("\n" * "=" ^ 60)
println("RESULTS SUMMARY")
println("=" ^ 60)
println("Adam (100 iters):              $(res_adam.objective)")
println("L-BFGS (100 iters):            $(res_lbfgs.objective)")
println("SciPy DE (200 iters):          $(sol_de.objective)")
println("SciPy Dual Annealing (200):    $(sol_da.objective)")
println("GPU-PSO (10k particles, 100):  $(gsol.cost)")
println("=" ^ 60)

