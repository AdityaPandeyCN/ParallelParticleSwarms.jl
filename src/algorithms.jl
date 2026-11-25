"""
    PSOAlgorithm

Abstract supertype for particle-swarm optimization algorithms implemented by
ParallelParticleSwarms.

# Interface

To add a CPU algorithm, subtype `PSOAlgorithm` and implement
[`pso_solve`](@ref) for the new type. The method must return
`(global_best, particles, solve_time)`, where `global_best` has `position` and
`cost` properties. ParallelParticleSwarms uses this interface to implement the
standard `SciMLBase.solve` contract for `OptimizationProblem`s.
"""
abstract type PSOAlgorithm end

"""
    pso_solve(prob, alg; maxiters, kwargs...) -> (global_best, particles, solve_time)

Run the algorithm-specific particle-swarm loop for a [`PSOAlgorithm`](@ref).

# Interface

Subtypes of `PSOAlgorithm` must implement this method. `global_best` must expose
`position` and `cost` properties, each entry of `particles` must expose a
`position` property, and the particle positions are stored as the solution's
`original` field. `solve_time` is the elapsed time in seconds.
"""
function pso_solve end
abstract type HybridPSOAlgorithm{LocalOpt} end
abstract type GPUSamplingAlgorithm end

struct GPUUniformSampler <: GPUSamplingAlgorithm
end

struct GPUUnboundedSampler <: GPUSamplingAlgorithm
end

"""
    ParallelPSOKernel(num_particles; global_update = true, backend = CPU(),
        θ = θ_default, γ = γ_default, h = sqrt, workgroupsize = 256)

Particle Swarm Optimization that launches a KernelAbstractions kernel for
parallel particle updates.

Static arrays for parameters in the `OptimizationProblem` are required for
successful GPU compilation.

# Arguments

- `num_particles`: Number of particles in the swarm.

# Keywords

- `global_update`: Whether particles share the global best position during each
    kernel update. Set to `false` to evolve particles independently.
- `backend`: KernelAbstractions backend used for computation.
- `θ`: PSO coefficient schedule.
- `γ`: PSO velocity update coefficient schedule.
- `h`: Transformation applied in the update rule.
- `workgroupsize`: KernelAbstractions workgroup size. Defaults to 256.
    With `global_update=true`, each block queues improving particles in shared
    memory and updates the global best under a lock.

# Examples

```julia
using KernelAbstractions
using ParallelParticleSwarms

alg = ParallelPSOKernel(100; backend = CPU(), global_update = false)
```

"""
struct ParallelPSOKernel{Backend, T, G, H} <: PSOAlgorithm
    num_particles::Int
    global_update::Bool
    backend::Backend
    θ::T
    γ::G
    h::H
    workgroupsize::Int
end

"""
    ParallelSyncPSOKernel(num_particles; backend = CPU(), θ = θ_default,
        γ = γ_default, h = sqrt, workgroupsize = 256, de_F = 0.0, de_CR = 0.9)

Particle Swarm Optimization that updates particles in parallel and synchronizes
after each generation to compute the global best position.

# Arguments

- `num_particles`: Number of particles in the swarm.

# Keywords

- `backend`: KernelAbstractions backend used for computation.
- `θ`: PSO coefficient schedule.
- `γ`: PSO velocity update coefficient schedule.
- `h`: Transformation applied in the update rule.
- `workgroupsize`: KernelAbstractions workgroup size. Defaults to 256.
- `de_F`: Differential-evolution step on the personal bests. When positive, every generation
    starts with each particle proposing `pbest[a] + de_F * (pbest[b] - pbest[c])` (three
    personal bests drawn at random, binomial crossover with its own personal best at rate
    `de_CR`), and keeping it as its personal best if it is no worse. This keeps the swarm's
    step size tied to how spread out the personal bests still are, so the swarm keeps
    contracting on rugged functions where plain global-best PSO stalls. Each step costs one
    extra objective evaluation per particle. Defaults to `0.0` (off).
- `de_CR`: Crossover rate of the differential-evolution step. Defaults to `0.9`.

# Examples

```julia
using KernelAbstractions
using ParallelParticleSwarms

alg = ParallelSyncPSOKernel(100; backend = CPU())
alg_de = ParallelSyncPSOKernel(100; backend = CPU(), de_F = 0.5)
```

"""
struct ParallelSyncPSOKernel{Backend, T, G, H} <: PSOAlgorithm
    num_particles::Int
    backend::Backend
    θ::T
    γ::G
    h::H
    workgroupsize::Int
    de_F::Float64
    de_CR::Float64
end

ParallelSyncPSOKernel(num_particles, backend, θ, γ, h, workgroupsize) =
    ParallelSyncPSOKernel(num_particles, backend, θ, γ, h, workgroupsize, 0.0, 0.9)

"""
    ParallelPSOArray(num_particles; θ = θ_default, γ = γ_default, h = sqrt)

Particle Swarm Optimization on a CPU. It keeps the arrays used in particle data structure
to be Julia's `Array`, which may be better for high-dimensional problems.

# Arguments

- `num_particles`: Number of particles in the swarm.

# Keywords

- `θ`: PSO coefficient schedule.
- `γ`: PSO velocity update coefficient schedule.
- `h`: Transformation applied in the update rule.

# Examples

```julia
using ParallelParticleSwarms

alg = ParallelPSOArray(100)
```

# Limitations

Running the optimization updates the global best positions with possible thread races.
This is the price to be paid to fuse all the updates into a single kernel. Techniques such as queue lock and atomic
updates can be used to fix this.

"""
struct ParallelPSOArray{T, G, H} <: PSOAlgorithm
    num_particles::Int
    θ::T
    γ::G
    h::H
end

"""
    SerialPSO(num_particles; θ = θ_default, γ = γ_default, h = sqrt)

Serial Particle Swarm Optimization on a CPU.

# Arguments

- `num_particles`: Number of particles in the swarm.

# Keywords

- `θ`: PSO coefficient schedule.
- `γ`: PSO velocity update coefficient schedule.
- `h`: Transformation applied in the update rule.

# Examples

```julia
using ParallelParticleSwarms

alg = SerialPSO(100)
```

"""
struct SerialPSO{T, G, H} <: PSOAlgorithm
    num_particles::Int
    θ::T
    γ::G
    h::H
end

function ParallelPSOKernel(
        num_particles::Int;
        global_update = true, backend = CPU(), θ = θ_default, γ = γ_default, h = sqrt,
        workgroupsize = 256
    )
    return ParallelPSOKernel(
        num_particles, global_update, backend, θ, γ, h, workgroupsize
    )
end

function ParallelSyncPSOKernel(
        num_particles::Int;
        backend = CPU(), θ = θ_default, γ = γ_default, h = sqrt,
        workgroupsize = 256, de_F = 0.0, de_CR = 0.9
    )
    de_F >= 0 || throw(ArgumentError("de_F must be non-negative, got $de_F"))
    0 <= de_CR <= 1 || throw(ArgumentError("de_CR must be in [0, 1], got $de_CR"))
    return ParallelSyncPSOKernel(
        num_particles, backend, θ, γ, h, workgroupsize, Float64(de_F), Float64(de_CR)
    )
end

function ParallelPSOArray(num_particles::Int; θ = θ_default, γ = γ_default, h = sqrt)
    return ParallelPSOArray(num_particles, θ, γ, h)
end

function SerialPSO(num_particles::Int; θ = θ_default, γ = γ_default, h = sqrt)
    return SerialPSO(num_particles, θ, γ, h)
end

SciMLBase.allowsbounds(::PSOAlgorithm) = true
SciMLBase.allowsconstraints(::PSOAlgorithm) = true

"""
    BFGS()

Local Broyden refinement used by [`HybridPSO`](@ref).
"""
struct BFGS end

"""
    HybridPSO(; backend = CPU(), pso = ParallelPSOKernel(100; global_update = false, backend),
        local_opt = SimpleLBFGS())

Run a particle-swarm search followed by local refinement.

# Keywords

- `backend`: KernelAbstractions backend used by the particle-swarm stage.
- `pso`: Particle-swarm algorithm used to generate local starting points.
- `local_opt`: `SimpleLBFGS` (projected Strong-Wolfe L-BFGS from
  SimpleOptimization.jl) or [`BFGS`](@ref) local refinement algorithm.
"""
struct HybridPSO{Backend, LocalOpt} <: HybridPSOAlgorithm{LocalOpt}
    pso::PSOAlgorithm
    local_opt::LocalOpt
    backend::Backend
end

function HybridPSO(;
        backend = CPU(),
        pso = ParallelParticleSwarms.ParallelPSOKernel(100; global_update = false, backend),
        local_opt = SimpleLBFGS()
    )
    return HybridPSO(pso, local_opt, backend)
end

SciMLBase.allowsbounds(::HybridPSOAlgorithm{LocalOpt}) where {LocalOpt} = true
