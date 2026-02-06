@inline function update_particle_state(particle, prob, gbest, w, c1, c2, iter, opt)
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

    update_pos = max.(particle.position, prob.lb)
    update_pos = min.(update_pos, prob.ub)
    @set! particle.position = update_pos

    particle = handle_constraints(particle, prob, iter, opt)

    if particle.cost < particle.best_cost
        @set! particle.best_position = particle.position
        @set! particle.best_cost = particle.cost
    end
    return particle
end

@inline function handle_constraints(particle, prob, iter, opt)
    if !isnothing(prob.f.cons)
        penalty = calc_penalty(particle.position, prob, iter + 1, opt.θ, opt.γ, opt.h)
        @set! particle.cost = prob.f(particle.position, prob.p) + penalty
    else
        @set! particle.cost = prob.f(particle.position, prob.p)
    end
    return particle
end

@kernel unsafe_indices = true function update_particle_states!(
        prob,
        gpu_particles::AbstractArray{SPSOParticle{T1, T2}}, gbest_ref, w,
        opt::ParallelPSOKernel, lock::AbstractArray{UInt32}; c1 = 1.4962f0,
        c2 = 1.4962f0
    ) where {T1, T2}
    tidx = @index(Local, Linear)
    gidx = @index(Group, Linear)

    @uniform gs = @groupsize()[1]
    n = length(gpu_particles)
    i = (gidx - 1) * gs + tidx

    best_queue = @localmem SPSOGBest{T1, T2} (gs)
    queue_num = @localmem UInt32 1

    particle = @private SPSOParticle{T1, T2} 1

    if i <= n
        @inbounds particle[1] = gpu_particles[i]
    end

    if tidx == 1
        fill!(
            best_queue,
            SPSOGBest(gbest_ref[1].position, convert(typeof(gbest_ref[1].cost), Inf))
        )
        queue_num[1] = UInt32(0)
    end

    @synchronize

    if i <= n
        @inbounds particle[1] = update_particle_state(
            particle[1],
            prob,
            gbest_ref[1],
            w,
            c1,
            c2,
            i,
            opt
        )
    end

    @synchronize

    if i <= n
        @inbounds if particle[1].best_cost < gbest_ref[1].cost
            queue_idx = @atomic queue_num[1] += UInt32(1)
            @inbounds best_queue[queue_idx] = SPSOGBest(
                particle[1].best_position,
                particle[1].best_cost
            )
        end
    end

    @synchronize

    if tidx == 1
        if queue_num[1] > 1
            for j in 2:queue_num[1]
                @inbounds if best_queue[j].cost < best_queue[1].cost
                    best_queue[1] = best_queue[j]
                end
            end

            while true
                res = @atomicreplace lock[1] UInt32(0) => UInt32(1)
                if res.success
                    break
                end
            end

            @inbounds if best_queue[1].cost < gbest_ref[1].cost
                gbest_ref[1] = best_queue[1]
            end

            while true
                res = @atomicreplace lock[1] UInt32(1) => UInt32(0)
                if res.success
                    break
                end
            end
        end
    end

    if i <= n
        @inbounds gpu_particles[i] = particle[1]
    end
end

@kernel function update_particle_states!(
        prob,
        gpu_particles::AbstractArray{SPSOParticle{T1, T2}}, gbest_ref, w,
        opt::ParallelPSOKernel{Backend, T, G, H}, lock::AbstractArray{UInt32}; c1 = 1.4962f0,
        c2 = 1.4962f0
    ) where {Backend <: CPU, T1, T2, T, G, H}
    i = @index(Global, Linear)

    @inbounds particle = gpu_particles[i]

    particle = update_particle_state(particle, prob, gbest_ref[1], w, c1, c2, i, opt)

    if particle.best_cost < gbest_ref[1].cost
        gbest_ref[1] = SPSOGBest(particle.best_position, particle.best_cost)
    end

    @inbounds gpu_particles[i] = particle
end

@kernel unsafe_indices = true function update_particle_states!(
        prob,
        gpu_particles::AbstractArray{SPSOParticle{T1, T2}}, block_particles, gbest, w,
        opt::ParallelSyncPSOKernel; c1 = 1.4962f0,
        c2 = 1.4962f0
    ) where {T1, T2}
    tidx = @index(Local, Linear)
    gidx = @index(Group, Linear)

    @uniform gs = @groupsize()[1]
    n = length(gpu_particles)
    i = (gidx - 1) * gs + tidx

    group_particles = @localmem SPSOGBest{T1, T2} (gs)

    if tidx == 1
        fill!(group_particles, SPSOGBest(gbest.position, convert(typeof(gbest.cost), Inf)))
    end

    @synchronize

    if i <= n
        @inbounds particle = gpu_particles[i]
        particle = update_particle_state(particle, prob, gbest, w, c1, c2, i, opt)
        @inbounds group_particles[tidx] = SPSOGBest(particle.best_position, particle.best_cost)
    end

    stride = div(gs, 2)

    while stride >= 1
        @synchronize
        if tidx <= stride
            @inbounds if group_particles[tidx].cost > group_particles[tidx + stride].cost
                group_particles[tidx] = group_particles[tidx + stride]
            end
        end
        stride = div(stride, 2)
    end

    @synchronize

    if tidx == 1
        @inbounds block_particles[gidx] = group_particles[tidx]
    end

    if i <= n
        @inbounds gpu_particles[i] = particle
    end
end

@kernel function update_particle_states!(
        prob, gpu_particles, gbest, w,
        opt::ParallelSyncPSOKernel{Backend, T, G, H}; c1 = 1.4962f0,
        c2 = 1.4962f0
    ) where {Backend <: CPU, T, G, H}
    i = @index(Global, Linear)

    @inbounds particle = gpu_particles[i]

    particle = update_particle_state(particle, prob, gbest, w, c1, c2, i, opt)

    @inbounds gpu_particles[i] = particle
end

@kernel unsafe_indices = true function update_particle_states_async!(
        prob,
        gpu_particles,
        gbest_ref,
        w, wdamp, maxiters, opt;
        c1 = 1.4962f0,
        c2 = 1.4962f0
    )
    tidx = @index(Local, Linear)
    gidx = @index(Group, Linear)
    @uniform gs = @groupsize()[1]
    i = (gidx - 1) * gs + tidx

    if i <= length(gpu_particles)
        gbest = gbest_ref[1]

        @inbounds particle = gpu_particles[i]

        for iter in 1:maxiters
            particle = update_particle_state(particle, prob, gbest, w, c1, c2, iter, opt)
            if particle.best_cost < gbest.cost
                @set! gbest.position = particle.best_position
                @set! gbest.cost = particle.best_cost
            end
            w = w * wdamp
        end

        @inbounds gpu_particles[i] = particle
        @inbounds gbest_ref[1] = gbest
    end
end

@kernel function update_particle_states_async!(
        prob,
        gpu_particles,
        gbest_ref,
        w, wdamp, maxiters, opt::ParallelPSOKernel{Backend, T, G, H};
        c1 = 1.4962f0,
        c2 = 1.4962f0
    ) where {Backend <: CPU, T, G, H}
    i = @index(Global, Linear)

    gbest = gbest_ref[1]

    @inbounds particle = gpu_particles[i]

    for iter in 1:maxiters
        particle = update_particle_state(particle, prob, gbest, w, c1, c2, iter, opt)
        if particle.best_cost < gbest.cost
            @set! gbest.position = particle.best_position
            @set! gbest.cost = particle.best_cost
        end
        w = w * wdamp
    end

    @inbounds gpu_particles[i] = particle
    @inbounds gbest_ref[1] = gbest
end
