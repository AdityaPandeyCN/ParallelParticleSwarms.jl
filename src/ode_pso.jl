using SciMLBase: ImmutableODEProblem
using KernelAbstractions
using Setfield

@kernel function ode_update_particle_states!(
        gpu_particles, lb, ub, gbest, w; c1 = 1.4962f0,
        c2 = 1.4962f0
    )
    i = @index(Global, Linear)
    if i <= length(gpu_particles)
        @inbounds particle = gpu_particles[i]

        updated_velocity = w .* particle.velocity .+
            c1 .* rand(typeof(particle.velocity)) .*
            (
            particle.best_position -
                particle.position
        ) .+
            c2 .* rand(typeof(particle.velocity)) .*
            (gbest.position - particle.position)

        @set! particle.velocity = updated_velocity

        @set! particle.position = particle.position + particle.velocity

        update_pos = max(particle.position, lb)
        update_pos = min(update_pos, ub)

        @set! particle.position = update_pos

        @inbounds gpu_particles[i] = particle
    end
end

@kernel function ode_update_particle_costs!(losses, gpu_particles)
    i = @index(Global, Linear)
    if i <= length(losses)
        @inbounds particle = gpu_particles[i]
        @inbounds loss = losses[i]

        @set! particle.cost = loss

        if particle.cost < particle.best_cost
            @set! particle.best_position = particle.position
            @set! particle.best_cost = particle.cost
        end

        @inbounds gpu_particles[i] = particle
    end
end

# ============================================================================
# GPU Kernels for remaking ImmutableODEProblems
# ============================================================================

# Simple case: p = position (parameters only, no tuple)
@kernel function _remake_probs_simple!(probs, improb, gpu_particles)
    i = @index(Global, Linear)
    if i <= length(gpu_particles)
        @inbounds position = gpu_particles[i].position
        @inbounds probs[i] = ImmutableODEProblem{false}(
            improb.f,
            improb.u0,
            improb.tspan,
            position
        )
    end
end

# Tuple case: p = (constant, position) - for neural ODEs where first element is fixed
@kernel function _remake_probs_tuple!(probs, improb, gpu_particles)
    i = @index(Global, Linear)
    if i <= length(gpu_particles)
        @inbounds position = gpu_particles[i].position
        @inbounds probs[i] = ImmutableODEProblem{false}(
            improb.f,
            improb.u0,
            improb.tspan,
            (improb.p[1], position)
        )
    end
end

# ============================================================================
# Trait-based dispatch for selecting the correct kernel
# ============================================================================

abstract type ParamStyle end
struct SimpleParams <: ParamStyle end
struct TupleParams <: ParamStyle end

# Auto-detect based on parameter type
param_style(::Tuple) = TupleParams()
param_style(_) = SimpleParams()

function _get_remake_kernel(backend, ::SimpleParams)
    return _remake_probs_simple!(backend)
end

function _get_remake_kernel(backend, ::TupleParams)
    return _remake_probs_tuple!(backend)
end

# ============================================================================
# Problem conversion utilities
# ============================================================================

# Convert ODEProblem to ImmutableODEProblem if needed
function ensure_immutable(prob::ODEProblem{iip}) where {iip}
    return ImmutableODEProblem{iip}(prob.f, prob.u0, prob.tspan, prob.p)
end

function ensure_immutable(prob::ImmutableODEProblem)
    return prob
end

# Legacy compatibility - if someone still uses make_prob_compatible
function make_prob_compatible(prob)
    return ensure_immutable(prob)
end

# ============================================================================
# Main parameter estimation functions
# ============================================================================

"""
    parameter_estim_ode!(prob, cache, lb, ub, Val(adaptive); kwargs...)

GPU-accelerated parameter estimation using Particle Swarm Optimization.

# Arguments
- `prob`: ODEProblem or ImmutableODEProblem to solve
- `cache`: NamedTuple with (losses, gpu_particles, gpu_data, gbest, probs, nn)
- `lb`: Lower bounds (StaticArray)
- `ub`: Upper bounds (StaticArray)
- `Val(true)`: Use adaptive solver (vectorized_asolve)
- `Val(false)`: Use fixed-step solver (vectorized_solve)

# Keyword Arguments
- `ode_alg`: ODE algorithm (default: GPUTsit5())
- `w`: Inertia weight (default: 0.7298f0)
- `wdamp`: Inertia damping factor (default: 1.0f0)
- `maxiters`: Maximum iterations (default: 100)
- Additional kwargs passed to ODE solver (saveat, dt, etc.)

# Returns
- `gbest`: Best solution found (SPSOGBest with position and cost)
"""
function parameter_estim_ode!(
        prob, cache,
        lb,
        ub, ::Val{true};
        ode_alg = GPUTsit5(),
        w = 0.7298f0,
        wdamp = 1.0f0,
        maxiters = 100, kwargs...
    )
    (losses, gpu_particles, gpu_data, gbest, probs) = cache
    backend = get_backend(gpu_particles)
    
    # Get kernels
    update_states! = ParallelParticleSwarms.ode_update_particle_states!(backend)
    update_costs! = ParallelParticleSwarms.ode_update_particle_costs!(backend)
    
    # Convert to ImmutableODEProblem and select appropriate remake kernel
    improb = ensure_immutable(prob)
    remake_probs! = _get_remake_kernel(backend, param_style(improb.p))
    
    n_particles = length(gpu_particles)

    for iter in 1:maxiters
        # Update particle positions and velocities
        update_states!(
            gpu_particles,
            lb,
            ub,
            gbest,
            w;
            ndrange = n_particles
        )
        KernelAbstractions.synchronize(backend)

        # Remake problems with new parameters (GPU kernel)
        remake_probs!(probs, improb, gpu_particles; ndrange = n_particles)
        KernelAbstractions.synchronize(backend)

        # Solve ODEs in parallel (adaptive)
        ts, us = vectorized_asolve(
            probs,
            prob,
            ode_alg; kwargs...
        )
        KernelAbstractions.synchronize(backend)

        # Compute losses
        sum!(losses, sum.((gpu_data .- us) .^ 2))

        # Update particle costs and personal bests
        update_costs!(losses, gpu_particles; ndrange = length(losses))
        KernelAbstractions.synchronize(backend)

        # Find global best
        best_particle = minimum(gpu_particles)
        KernelAbstractions.synchronize(backend)

        gbest = ParallelParticleSwarms.SPSOGBest(
            best_particle.best_position, best_particle.best_cost
        )
        w = w * wdamp
    end
    return gbest
end

function parameter_estim_ode!(
        prob, cache,
        lb,
        ub, ::Val{false};
        ode_alg = GPUTsit5(),
        w = 0.7298f0,
        wdamp = 1.0f0,
        maxiters = 100, kwargs...
    )
    (losses, gpu_particles, gpu_data, gbest, probs) = cache
    backend = get_backend(gpu_particles)
    
    # Get kernels
    update_states! = ParallelParticleSwarms.ode_update_particle_states!(backend)
    update_costs! = ParallelParticleSwarms.ode_update_particle_costs!(backend)
    
    # Convert to ImmutableODEProblem and select appropriate remake kernel
    improb = ensure_immutable(prob)
    remake_probs! = _get_remake_kernel(backend, param_style(improb.p))
    
    n_particles = length(gpu_particles)

    for iter in 1:maxiters
        # Update particle positions and velocities
        update_states!(
            gpu_particles,
            lb,
            ub,
            gbest,
            w;
            ndrange = n_particles
        )
        KernelAbstractions.synchronize(backend)

        # Remake problems with new parameters (GPU kernel)
        remake_probs!(probs, improb, gpu_particles; ndrange = n_particles)
        KernelAbstractions.synchronize(backend)

        # Solve ODEs in parallel (fixed-step)
        ts, us = vectorized_solve(
            probs,
            prob,
            ode_alg; kwargs...
        )
        KernelAbstractions.synchronize(backend)

        # Compute losses
        sum!(losses, sum.((gpu_data .- us) .^ 2))

        # Update particle costs and personal bests
        update_costs!(losses, gpu_particles; ndrange = length(losses))
        KernelAbstractions.synchronize(backend)

        # Find global best
        best_particle = minimum(gpu_particles)
        KernelAbstractions.synchronize(backend)

        gbest = ParallelParticleSwarms.SPSOGBest(
            best_particle.best_position, best_particle.best_cost
        )
        w = w * wdamp
    end
    return gbest
end

# ============================================================================
# Convenience function for cache initialization
# ============================================================================

"""
    init_ode_pso_cache(prob, particles, data, gbest, backend)

Initialize cache for parameter_estim_ode!

# Arguments
- `prob`: ImmutableODEProblem or ODEProblem
- `particles`: CPU array of particles
- `data`: Training data (will be converted to GPU SVector array)
- `gbest`: Initial global best
- `backend`: GPU backend (e.g., CUDABackend())

# Returns
- NamedTuple cache ready for parameter_estim_ode!
"""
function init_ode_pso_cache(prob, particles, data, gbest, backend)
    n_particles = length(particles)
    
    # Adapt particles to GPU
    gpu_particles = adapt(backend, particles)
    
    # Create losses array
    losses = adapt(backend, ones(eltype(prob.u0), (1, n_particles)))
    
    # Convert data to GPU SVector array
    gpu_data = adapt(
        backend,
        [
            SVector{size(data, 1), eltype(data)}(@view data[:, i])
            for i in 1:size(data, 2)
        ]
    )
    
    # Pre-allocate problems array
    improb = ensure_immutable(prob)
    probs = adapt(backend, fill(improb, n_particles))
    
    return (; losses, gpu_particles, gpu_data, gbest, probs)
end

