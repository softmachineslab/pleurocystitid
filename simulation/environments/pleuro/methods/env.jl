

"""
    Pleuro <: Environment

    extinct crinoid with a single motile appendage
"""
struct Pleuro end

function pleuro(; 
    representation=:minimal, 
    timestep=0.05, 
    gravity=[0.0; 0.0; -9.81],
    friction_coefficient=0.5, 
    spring=0.0, 
    damper=1.0, 
    seed=1,
    contact_feet=true, 
    contact_body=true,
    limits::Bool=true,
    info=nothing, 
    vis=Visualizer(), 
    name=:robot,
    opts_step=SolverOptions(), 
    opts_grad=SolverOptions(),
    num_links=2,
    num_u=1,
    dim3=false,
    radius=0.2/20,
    radius_b=1/20,
    height=0.5/20,
    T=Float64)

    mechanism = get_pleuro(
        timestep=timestep, 
        gravity=gravity, 
        friction_coefficient=friction_coefficient, 
        spring=spring, 
        damper=damper, 
        contact_feet=contact_feet, 
        contact_body=contact_body, 
        contact_type=:nonlinear,
        num_links=num_links,
        num_u=num_u,
        dim3=dim3,
        radius=radius,
        radius_b=radius_b,
        height=height,
        limits=limits)

    initialize_pleuro!(mechanism)

    if representation == :minimal
        nx = minimal_dimension(mechanism)
    elseif representation == :maximal
        nx = maximal_dimension(mechanism)
    end

    if dim3
        nu = num_u*2
        total_joints = num_links*2
    else
        nu = num_u
        total_joints = num_links
    end
    no = nx + length(mechanism.contacts)

    aspace = BoxSpace(nu, 
        low=(-ones(nu)), 
        high=(ones(nu)))
    ospace = BoxSpace(no, 
        low=(-Inf * ones(no)), 
        high=(Inf * ones(no)))

    rng = MersenneTwister(seed)

    z = get_maximal_state(mechanism)
    x = representation == :minimal ? maximal_to_minimal(mechanism, z) : z

    fx = zeros(nx, nx)
    fu = zeros(nx, nu)

    u_prev = zeros(nu)
    control_mask = [zeros(nu, 8) I(nu) zeros(nu,total_joints-nu-2)]
    control_scaling = Diagonal(timestep * 150.0 * ones(nu))
    Dojo.build_robot(mechanism, vis=vis, name=name)

    TYPES = [Pleuro, T, typeof(mechanism), typeof(aspace), typeof(ospace), typeof(info)]
    env = Environment{TYPES...}(mechanism, representation, aspace, ospace,
        x, fx, fu,
        u_prev, 
        control_mask' * control_scaling,
        nx, nu, no,
        info,
        [rng], vis,
        opts_step, opts_grad)
    return env
end

function Base.step(env::Environment{Pleuro}, x, u; 
    gradients=false,
    attitude_decompress=false)

    mechanism = env.mechanism
    timestep= mechanism.timestep

    # x position (before)
    xposbefore = x[1]

    x0 = copy(x)
    env.input_previous .= u  # for rendering in Gym
	u_scaled = env.control_map * u

    z0 = env.representation == :minimal ? minimal_to_maximal(mechanism, x0) : x0

    z1 = Dojo.step!(mechanism, z0, u_scaled; opts=env.opts_step)
    env.state .= env.representation == :minimal ? maximal_to_minimal(mechanism, z1) : z1

    # x position (after)
    xposafter = env.state[1]

    # forward reward
    forward_reward = 2.0 * (xposafter - xposbefore) / timestep

    # control cost
	ctrl_cost = (0.05 * u' * u)[1]

    # contact cost
    contact_cost = 0.0

    for contact in mechanism.contacts
        contact_cost += 0.5 * 1.0e-3 * max(-1.0, min(1.0, contact.impulses[2][1]))^2.0
    end

	# survive_reward = 1.0
    survive_reward = 0.05

    # total reward
    reward = forward_reward - ctrl_cost - contact_cost + survive_reward

    # done ?
    done = !(all(isfinite.(env.state)) && (env.state[3] >= 0.2) && (env.state[3] <= 1.0))

    # Gradients
    if gradients
        if env.representation == :minimal
            fx, fu = Dojo.get_minimal_gradients!(env.mechanism, z0, u_scaled, opts=env.opts_grad)
        elseif env.representation == :maximal
            fx, fu = Dojo.get_maximal_gradients!(env.mechanism, z0, u_scaled, opts=env.opts_grad)
            if attitude_decompress 
                if attitude_decompress
                    A0 = attitude_jacobian(z0, length(env.mechanism.bodies))
                    A1 = attitude_jacobian(z1, length(env.mechanism.bodies))
                    fx = A1 * fx * A0'
                    fu = A1 * fu
                end
            end
        end
        env.dynamics_jacobian_state .= fx
        env.dynamics_jacobian_input .= fu * env.control_map
    end

    info = Dict()

    return get_observation(env), reward, done, info
end

# TODO add random noise
function Base.reset(env::Environment{Pleuro}; 
    x=nothing)
    # include(joinpath(@__DIR__, "initialize.jl"))
    #@show varinfo()
    initialize_pleuro!(env.mechanism)
    #initialize!(env.mechanism, Dojo.type2symbol(Patrick))

    if x != nothing
        env.state .= x
    else
        x = get_minimal_state(env.mechanism)
        if env.representation == :minimal
            set_maximal_state!(env.mechanism, minimal_to_maximal(env.mechanism, x))
            env.state .= x
        elseif env.representation == :maximal
            z = minimal_to_maximal(env.mechanism, x)
            set_maximal_state!(env.mechanism, z)
            env.state .= z
        end
        env.input_previous .= 0.0
    end

    return get_observation(env)
end

function get_observation(env::Environment{Pleuro,T}) where T
    contact_force = T[]
    for contact in env.mechanism.contacts
        push!(contact_force, max(-1.0, min(1.0, contact.impulses[2][1])))
    end
    return [env.state; contact_force]
end

function cost(env::Environment{Pleuro}, x, u;
    forward_reward_weight=1.0, 
    ctrl_cost_weight=0.001)

    x_velocity = -x[7]
    #@show x_velocity
    # @show mean(abs.(u))
    c = ctrl_cost_weight * u' * u - x_velocity * forward_reward_weight
    return c
end
