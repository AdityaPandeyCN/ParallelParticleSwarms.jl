using ParallelParticleSwarms, Optimization, SciMLBase, StaticArrays, KernelAbstractions, Test

@testset "SimpleLBFGS hybrid local polish" begin
    function _solve_hybrid(
            f, x0, p = nothing; lb = nothing, ub = nothing,
            maxiters = 50, local_maxiters = 200, adtype = Optimization.AutoForwardDiff()
        )
        n = length(x0)
        optf = OptimizationFunction{false}(f, adtype)
        u0 = SVector{n, Float64}(x0)
        kwargs = NamedTuple()
        if lb !== nothing
            kwargs = (;
                lb = SVector{n, Float64}(lb),
                ub = SVector{n, Float64}(ub),
            )
        end
        return solve(
            OptimizationProblem(optf, u0, p; kwargs...),
            HybridPSO(;
                pso = ParallelSyncPSOKernel(64; backend = CPU()),
                backend = CPU(),
                local_opt = SimpleLBFGS(),
            );
            maxiters, local_maxiters, abstol = 1.0e-10
        )
    end

    beale(x, p) = (1.5 - x[1] + x[1] * x[2])^2 +
        (2.25 - x[1] + x[1] * x[2]^2)^2 +
        (2.625 - x[1] + x[1] * x[2]^3)^2
    booth(x, p) = (x[1] + 2x[2] - 7)^2 + (2x[1] + x[2] - 5)^2
    himmelblau(x, p) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    quadratic(x, p) = p[1] * (x[1] - 1)^2 + (x[2] + 2)^2
    boundary_quadratic(x, p) = (x[1] + one(eltype(x)))^2
    rosen2(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

    sol = _solve_hybrid(rosen2, [0.0, 0.0], [1.0, 100.0])
    @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
    @test sol.objective < 1.0e-8

    sol = _solve_hybrid(rosen2, [-1.2, 1.0], [1.0, 100.0])
    @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
    @test sol.objective < 1.0e-8

    sol = _solve_hybrid(rosen2, [0.0, 0.0], [1.0, 100.0]; lb = [-2.0, -2.0], ub = [2.0, 2.0])
    @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
    @test sol.objective < 1.0e-8
    @test all(-2 .≤ sol.u) && all(sol.u .≤ 2)

    @testset "chunked ForwardDiff gradient, chunksize = $K" for K in (1, 3, 16)
        f(x, p) = sum(i -> p[i] * x[i]^2 * x[mod1(i + 1, 10)], 1:10) + prod(x) / 10
        x = SVector{10}(range(0.5, 1.4; length = 10))
        p = SVector{10}(1.0:10.0)
        g = ParallelParticleSwarms.instantiate_gradient(f, Optimization.AutoForwardDiff(; chunksize = K))
        @test g isa ParallelParticleSwarms.ChunkedForwardDiffGradient{K}
        @test g(x, p) isa SVector{10, Float64}
        @test g(x, p) ≈ ParallelParticleSwarms.ForwardDiff.gradient(y -> f(y, p), x)
    end

    adtypes = (Optimization.AutoEnzyme(), SciMLBase.NoAD(), Optimization.AutoForwardDiff(; chunksize = 1))
    @testset "adtype = $(nameof(typeof(adtype)))" for adtype in adtypes
        sol = _solve_hybrid(rosen2, [-1.2, 1.0], [1.0, 100.0]; adtype)
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
        @test sol.objective < 1.0e-8

        sol = _solve_hybrid(
            rosen2, [0.0, 0.0], [1.0, 100.0]; adtype, lb = [-2.0, -2.0], ub = [2.0, 2.0]
        )
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
        @test sol.objective < 1.0e-8
    end

    sol = _solve_hybrid(beale, [1.0, 1.0])
    @test sol.u ≈ [3.0, 0.5] atol = 1.0e-4
    @test sol.objective < 1.0e-8

    sol = _solve_hybrid(booth, [1.0, 1.0])
    @test sol.u ≈ [1.0, 3.0] atol = 1.0e-4
    @test sol.objective < 1.0e-8

    sol = _solve_hybrid(himmelblau, [1.0, 1.0])
    @test sol.objective < 1.0e-8

    sol = _solve_hybrid(quadratic, [4.0, 4.0], [1000.0]; lb = [-5.0, -5.0], ub = [5.0, 5.0])
    @test sol.u ≈ [1.0, -2.0] atol = 1.0e-4
    @test sol.objective < 1.0e-8
    @test all(-5 .≤ sol.u) && all(sol.u .≤ 5)

    sol = _solve_hybrid(
        boundary_quadratic, [1.5]; lb = [0.0], ub = [2.0],
        maxiters = 20, local_maxiters = 50
    )
    @test sol.u[1] ≈ 0.0 atol = 1.0e-6
    @test sol.objective ≈ 1.0 atol = 1.0e-6
end
