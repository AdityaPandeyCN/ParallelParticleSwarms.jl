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

function default_prob_func(prob, gpu_particle)
return remake(prob, p = gpu_particle.position)
end

function parameter_estim_ode!(
    prob::ODEProblem, cache,
    lb,
    ub, ::Val{true};
    ode_alg = GPUTsit5(),
    prob_func = default_prob_func,
    w = 0.7298f0,
    wdamp = 1.0f0,
    maxiters = 100, kwargs...
)
(losses, gpu_particles, gpu_data, gbest) = cache
backend = get_backend(gpu_particles)
update_states! = ParallelParticleSwarms.ode_update_particle_states!(backend)
update_costs! = ParallelParticleSwarms.ode_update_particle_costs!(backend)
data_host = Array(gpu_data)
loss_values = Vector{Float32}(undef, length(gpu_particles))

    improb = make_prob_compatible(prob)
    probs = Vector{typeof(improb)}(undef, length(gpu_particles))

for i in 1:maxiters
    update_states!(
        gpu_particles,
        lb,
        ub,
        gbest,
        w;
        ndrange = length(gpu_particles)
    )

    KernelAbstractions.synchronize(backend)

    host_particles = Array(gpu_particles)
    @inbounds for (idx, particle) in enumerate(host_particles)
        probs[idx] = prob_func(improb, particle)
    end

    KernelAbstractions.synchronize(backend)

    ###TODO: Somehow vectorized_asolve hangs and does not here :(

    ts, us = vectorized_asolve(
        probs,
        improb,
        ode_alg; kwargs...
    )

    KernelAbstractions.synchronize(backend)

    host_us = Array(us)
    @inbounds for p in 1:length(gpu_particles)
        loss_values[p] = sum(
            sum(abs2, data_host[k] - host_us[k, p])
            for k in 1:length(data_host)
        )
    end
    copyto!(losses, loss_values)

    update_costs!(losses, gpu_particles; ndrange = length(losses))

    KernelAbstractions.synchronize(backend)

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
    prob::ODEProblem, cache,
    lb,
    ub, ::Val{false};
    ode_alg = GPUTsit5(),
    prob_func = default_prob_func,
    w = 0.7298f0,
    wdamp = 1.0f0,
    maxiters = 100, kwargs...
)
(losses, gpu_particles, gpu_data, gbest) = cache
backend = get_backend(gpu_particles)
update_states! = ParallelParticleSwarms.ode_update_particle_states!(backend)
update_costs! = ParallelParticleSwarms.ode_update_particle_costs!(backend)
data_host = Array(gpu_data)
loss_values = Vector{Float32}(undef, length(gpu_particles))

    improb = make_prob_compatible(prob)
    probs = Vector{typeof(improb)}(undef, length(gpu_particles))

for i in 1:maxiters
    update_states!(
        gpu_particles,
        lb,
        ub,
        gbest,
        w;
        ndrange = length(gpu_particles)
    )

    KernelAbstractions.synchronize(backend)

    host_particles = Array(gpu_particles)
    @inbounds for (idx, particle) in enumerate(host_particles)
        probs[idx] = prob_func(improb, particle)
    end

    KernelAbstractions.synchronize(backend)

    ###TODO: Somehow vectorized_asolve hangs and does not here :(

    ts, us = vectorized_solve(
        probs,
        improb,
        ode_alg; kwargs...
    )

    KernelAbstractions.synchronize(backend)

    host_us = Array(us)
    @inbounds for p in 1:length(gpu_particles)
        loss_values[p] = sum(
            sum(abs2, data_host[k] - host_us[k, p])
            for k in 1:length(data_host)
        )
    end
    copyto!(losses, loss_values)

    update_costs!(losses, gpu_particles; ndrange = length(losses))

    KernelAbstractions.synchronize(backend)

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
    prob::SciMLBase.ImmutableODEProblem, cache,
    lb,
    ub, ::Val{B}; ode_alg = GPUTsit5(), prob_func = default_prob_func,
    w = 0.7298f0, wdamp = 1.0f0, maxiters = 100, kwargs...
) where {B}
    mutable_prob = ODEProblem(prob.f, prob.u0, prob.tspan, prob.p)
    return parameter_estim_ode!(
        mutable_prob,
        cache,
        lb,
        ub,
        Val{B}();
        ode_alg = ode_alg,
        prob_func = prob_func,
        w = w,
        wdamp = wdamp,
        maxiters = maxiters,
        kwargs...
    )
end

