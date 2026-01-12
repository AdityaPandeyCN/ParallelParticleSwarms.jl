using Pkg
Pkg.activate(@__DIR__)

using SimpleChains,
    StaticArrays, OrdinaryDiffEq, SciMLSensitivity, Optimization
using OptimizationOptimisers
using Optimisers: Adam
using OptimizationOptimJL
using OptimizationSciPy

using ParallelParticleSwarms
using CUDA
using KernelAbstractions
using Adapt
using Random
using Setfield

device!(0)

println("=" ^ 60)
println("NEURAL ODE BENCHMARK")
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

function loss_scipy(u, _)
    u32 = Float32.(u)
    pred = Array(solve(sprob_nn, Tsit5(); p = u32, saveat = tsteps))
    return sum(abs2, data .- pred)
end

lb_scipy = fill(-10.0, length(p_nn))
ub_scipy = fill(10.0, length(p_nn))
x0_scipy = Float64.(p_nn)
scipy_prob = OptimizationProblem(loss_scipy, x0_scipy; lb = lb_scipy, ub = ub_scipy)

@info "SciPy DE (maxiters=200)"
@time sol_de = Optimization.solve(scipy_prob, OptimizationSciPy.ScipyDifferentialEvolution(), maxiters = 200)
@show sol_de.objective

@info "SciPy DualAnnealing (maxiters=200)"
@time sol_da = Optimization.solve(scipy_prob, OptimizationSciPy.ScipyDualAnnealing(), maxiters = 200)
@show sol_da.objective

## GPU-PSO

function nn_fn(u::T, p, t)::T where {T}
    nn, ps = p
    return nn(u, ps)
end

p_static = SArray{Tuple{size(p_nn)...}}(p_nn...)

prob_nn = ODEProblem{false, SciMLBase.FullSpecialize}(nn_fn, u0, tspan, (sc, p_static))

n_particles = 10_000
backend = CUDABackend()

function loss_pso(u, p)
    odeprob, t = p
    _prob = remake(odeprob; p = (odeprob.p[1], u))
    pred = Array(solve(_prob, Tsit5(), saveat = t))
    return sum(abs2, data .- pred)
end

lb = @SArray fill(Float32(-10.0), length(p_static))
ub = @SArray fill(Float32(10.0), length(p_static))

soptprob = OptimizationProblem(loss_pso, prob_nn.p[2], (prob_nn, tsteps); lb = lb, ub = ub)

Random.seed!(rng, 0)
opt = ParallelPSOKernel(n_particles)
gbest, particles = ParallelParticleSwarms.init_particles(soptprob, opt, typeof(p_static))

gpu_data = adapt(
    backend,
    [
        SVector{length(prob_nn.u0), eltype(prob_nn.u0)}(@view data[:, i])
            for i in 1:length(tsteps)
    ]
)

CUDA.allowscalar(false)

gpu_particles = adapt(backend, particles)
losses = adapt(backend, ones(eltype(prob.u0), (1, n_particles)))

# Create probs array - use remake to get correct type
improb = ParallelParticleSwarms.make_prob_compatible(prob_nn)
init_prob = remake(improb, p = (sc, particles[1].position))
probs = CuArray(fill(init_prob, n_particles))

# Cache: 6 elements (includes probs and nn for kernel)
solver_cache = (; losses, gpu_particles, gpu_data, gbest, probs, nn = sc)

@info "GPU-PSO Warmup (compilation)"
@time gsol = ParallelParticleSwarms.parameter_estim_ode!(
    prob_nn,
    solver_cache, lb, ub, Val(true);
    saveat = tsteps, dt = 0.1f0, maxiters = 10
)

# Reset for fair benchmark
Random.seed!(rng, 0)
gbest, particles = ParallelParticleSwarms.init_particles(soptprob, opt, typeof(p_static))
gpu_particles = adapt(backend, particles)
losses = adapt(backend, ones(eltype(prob.u0), (1, n_particles)))
init_prob = remake(improb, p = (sc, particles[1].position))
probs = CuArray(fill(init_prob, n_particles))
solver_cache = (; losses, gpu_particles, gpu_data, gbest, probs, nn = sc)

@info "GPU-PSO (n_particles=$n_particles, maxiters=100)"
@time gsol = ParallelParticleSwarms.parameter_estim_ode!(
    prob_nn,
    solver_cache, lb, ub, Val(true);
    saveat = tsteps, dt = 0.1f0, maxiters = 100
)

@show gsol.cost

println("\n" * "=" ^ 60)
println("RESULTS SUMMARY")
println("=" ^ 60)
println("Adam:                $(res_adam.objective)")
println("L-BFGS:              $(res_lbfgs.objective)")
println("SciPy DE:            $(sol_de.objective)")
println("SciPy DualAnnealing: $(sol_da.objective)")
println("GPU-PSO:             $(gsol.cost)")
println("=" ^ 60)
