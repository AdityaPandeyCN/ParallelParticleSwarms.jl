using ParallelParticleSwarms, Optimization, SciMLBase, StaticArrays, KernelAbstractions, Test
using Random

# Rastrigin: many regular local minima, global minimum 0 at the origin.
rastrigin(x, p) = 10 * length(x) + sum(xi -> xi^2 - 10 * cos(2π * xi), x)
sphere(x, p) = sum(abs2, x)

function _prob(f, D; cons = nothing)
    lb = SVector{D, Float64}(ntuple(_ -> -5.12, D))
    ub = SVector{D, Float64}(ntuple(_ -> 5.12, D))
    optf = cons === nothing ? OptimizationFunction{false}(f) :
        OptimizationFunction{false}(f; cons)
    return OptimizationProblem{false}(optf, SVector{D, Float64}(ntuple(_ -> 1.0, D)), nothing; lb, ub)
end

@testset "ParallelSyncPSOKernel keyword validation" begin
    @test_throws ArgumentError ParallelSyncPSOKernel(10; de_F = -0.1)
    @test_throws ArgumentError ParallelSyncPSOKernel(10; de_CR = 1.5)
    @test_throws ArgumentError ParallelSyncPSOKernel(10; island_size = -1)
    alg = ParallelSyncPSOKernel(10)
    @test alg.de_F == 0.0 && alg.de_CR == 0.9 && alg.island_size == 0
    # The positional constructor without the new fields keeps the old behaviour.
    old = ParallelSyncPSOKernel(10, CPU(), alg.θ, alg.γ, alg.h, 256)
    @test old.de_F == 0.0 && old.island_size == 0
end

@testset "DE step and islands on the CPU backend" begin
    Random.seed!(1)
    prob = _prob(rastrigin, 4)
    for alg in (
            ParallelSyncPSOKernel(200; backend = CPU(), de_F = 0.5),
            ParallelSyncPSOKernel(400; backend = CPU(), de_F = 0.5, island_size = 50),
            # Last island smaller than the others.
            ParallelSyncPSOKernel(430; backend = CPU(), de_F = 0.5, island_size = 100),
        )
        sol = solve(prob, alg; maxiters = 1500)
        @test sol.objective < 1.0e-8
        @test all(prob.lb .<= sol.u .<= prob.ub)
    end

    # Islands without the DE step still converge on a smooth problem.
    sol = solve(_prob(sphere, 3), ParallelSyncPSOKernel(300; backend = CPU(), island_size = 30); maxiters = 400)
    @test sol.objective < 1.0e-8
end

@testset "Personal and island bests never get worse" begin
    Random.seed!(2)
    prob = _prob(rastrigin, 5)
    alg = ParallelSyncPSOKernel(250; backend = CPU(), de_F = 0.5, island_size = 50)
    cache = init(prob, alg)
    best = map(p -> p.best_cost, cache.particles)
    obj = Inf
    for _ in 1:20
        sol = SciMLBase.solve!(cache; maxiters = 10)
        now = map(p -> p.best_cost, cache.particles)
        @test all(now .<= best)
        @test sol.objective <= obj
        # The reported best is the best personal best over all islands.
        @test sol.objective ≈ minimum(now)
        best, obj = now, sol.objective
    end
end

@testset "The DE step keeps an island's draws inside the island" begin
    # Island 1 sits at +3, island 2 at -3, every particle at its island's point: a trial built
    # only from the particle's own island lands on that point, so no particle may move.
    D = 2
    prob = _prob(sphere, D)
    alg = ParallelSyncPSOKernel(8; backend = CPU(), de_F = 0.5, island_size = 4)
    cache = init(prob, alg)
    T1 = typeof(prob.u0)
    for i in 1:8
        x = T1(ntuple(_ -> i <= 4 ? 3.0 : -3.0, D))
        c = sphere(x, nothing)
        cache.particles[i] = ParallelParticleSwarms.SPSOParticle(x, zero(T1), c, x, c)
    end
    snapshot = similar(cache.particles, T1)
    for it in 1:50
        ParallelParticleSwarms.run_de_step!(prob, cache.particles, snapshot, alg, it, 8)
    end
    @test all(p -> p.best_position == T1(3.0, 3.0), cache.particles[1:4])
    @test all(p -> p.best_position == T1(-3.0, -3.0), cache.particles[5:8])
end

@testset "DE step with constraints" begin
    # The problem and feasibility check of test/constraints.jl, with the DE step and islands on:
    # the DE trial goes through the same penalty as the PSO move.
    Random.seed!(1234)
    objective(x, p) = (x[1] - p[1])^2 + (x[2] - p[2])^2
    conss(x, p) = SVector{3}(-x[1] + 2 * x[2] - 1, +x[1] - 2 * x[2] + 1, (x[1]^2) / 4 + x[2]^2 - 1)
    prob = OptimizationProblem(
        OptimizationFunction{false}(objective, cons = conss), SVector(1.0f0, 1.0f0),
        SVector(2.0f0, 1.0f0); lcons = SVector(-Inf32, -Inf32), ucons = SVector(0.0f0, 0.0f0),
        lb = SVector(0.0f0, 0.0f0), ub = SVector(2.0f0, 2.0f0)
    )
    for kw in ((; de_F = 0.5), (; de_F = 0.5, island_size = 100))
        sol = solve(prob, ParallelSyncPSOKernel(1000; backend = CPU(), kw...); maxiters = 500)
        @test sol.retcode == ReturnCode.Default
        @test abs(1 - 2 * sol.u[2] + sol.u[1]) < 1.0e-1
    end
end

@testset "HybridPSO with a DE island swarm" begin
    Random.seed!(4)
    prob = OptimizationProblem{false}(
        OptimizationFunction{false}(rastrigin, Optimization.AutoForwardDiff()),
        SVector{4, Float64}(ntuple(_ -> 1.0, 4)), nothing;
        lb = SVector{4, Float64}(ntuple(_ -> -5.12, 4)), ub = SVector{4, Float64}(ntuple(_ -> 5.12, 4))
    )
    alg = HybridPSO(;
        pso = ParallelSyncPSOKernel(400; backend = CPU(), de_F = 0.5, island_size = 50),
        backend = CPU(),
    )
    sol = solve(prob, alg; maxiters = 800, local_maxiters = 50)
    @test sol.objective < 1.0e-8
end
