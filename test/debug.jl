using CUDA, ParallelParticleSwarms, StaticArrays, Optimization, ForwardDiff
using SimpleNonlinearSolve, KernelAbstractions, SciMLBase

function rosenbrock(x, p)
    sum(100.0f0 * (x[i+1] - x[i]^2)^2 + (1.0f0 - x[i])^2 for i in 1:(length(x)-1))
end

N = 10
x0 = @SVector zeros(Float32, N)
lb = @SVector fill(-10.0f0, N)
ub = @SVector fill(10.0f0, N)
p  = @SVector Float32[1.0, 100.0]

prob = OptimizationProblem(OptimizationFunction(rosenbrock, AutoForwardDiff()), x0, p; lb, ub)

# PSO phase
hybrid_opt = ParallelParticleSwarms.HybridPSO(CUDABackend(), ParallelParticleSwarms.LBFGS())
cache = ParallelParticleSwarms.init(prob, hybrid_opt)
sol_pso = solve!(cache.pso_cache)
x0s = sol_pso.original
CUDA.device_synchronize()
println("PSO done: $(length(x0s)) particles")

# Setup BFGS kernel
prob_nb = remake(prob, lb=nothing, ub=nothing)
∇f = ParallelParticleSwarms.instantiate_gradient(prob_nb.f.f, prob_nb.f.adtype)
nlprob = SimpleNonlinearSolve.ImmutableNonlinearProblem{false}(∇f, prob.u0, prob.p)
nlalg = SimpleLimitedMemoryBroyden(; threshold=10, linesearch=Val(true))
result = cache.start_points
copyto!(result, x0s)

# Test kernel at increasing scale
kernel = ParallelParticleSwarms.simplebfgs_run!(CUDABackend())
for n in [1, 32, 128, 256, 512, 1000]
    n > length(x0s) && continue
    copyto!(result, x0s)
    CUDA.device_synchronize()
    try
        kernel(nlprob, x0s, result, nlalg, 10, nothing, nothing; ndrange=n)
        CUDA.device_synchronize()
        println("ndrange=$n ✓")
    catch e
        println("ndrange=$n ✗ — $(sprint(showerror, e))")
        break
    end
end

# Test post-kernel broadcast
try
    sol_bfgs = (x -> prob_nb.f(x, prob.p)).(result)
    CUDA.device_synchronize()
    println("Broadcast ✓")
catch e
    println("Broadcast ✗ — $(sprint(showerror, e))")
end