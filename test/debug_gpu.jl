# test/debug_gpu.jl
using CUDA
using ParallelParticleSwarms, StaticArrays, SciMLBase, Test, LinearAlgebra, Random

backend = CUDABackend()

Random.seed!(1234)

N = 2
lb = @SArray ones(Float32, N)
lb = -1 * lb
ub = @SArray fill(Float32(10.0), N)

function rosenbrock(x, p)
    res = zero(eltype(x))
    for i in 1:(length(x) - 1)
        res += p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2
    end
    res
end

x0 = @SArray zeros(Float32, N)
p = @SArray Float32[1.0, 100.0]

prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

n_particles = 5000

println("Testing ParallelSyncPSOKernel...")
sol = solve(
    prob,
    ParallelSyncPSOKernel(n_particles; backend),
    maxiters = 500
)
println("Result: $(sol.objective)")