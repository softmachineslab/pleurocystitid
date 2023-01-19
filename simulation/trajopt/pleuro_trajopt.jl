using Pkg
Pkg.activate(joinpath(@__DIR__, "../examples/"))
Pkg.add("MAT")
Pkg.instantiate()

# ## Setup
using Dojo
using Random
using LinearAlgebra 
using JLD2
using StaticArrays
using IterativeLQR
using Dates
using MAT
using Plots
import Dojo: cost



#include(joinpath(@__DIR__, "algorithms/ags.jl")) # augmented random search
include(joinpath(@__DIR__, "../environments/pleuro/methods/initialize.jl"))
include(joinpath(@__DIR__, "../environments/pleuro/methods/env.jl"))
# ## Ant
num_links = 5
num_u = 2
dim3 = true
timestep=0.05
scale=1
factor = 20/scale
radius = 0.01*scale
radius_b = 0.05*scale
h_total = 0.15*scale
height = h_total/num_links
height_b = 2*radius
density = 1500
mass = π*radius^2*height*density
mass_2 = mass
mass_b = π*radius_b^2*height_b*density
mass_total = (mass*num_u + mass_2*(num_links-num_u) + mass_b)
Fg = 9.81*mass_total
V_b = π*radius^2*height*num_links + π*radius_b^2*height_b
Fb = V_b*1000*9.81
# g = 9.81
g = (Fg-Fb)/mass_total
@show g
gravity=[0.0; 0.0; -g]
env = pleuro(
    representation=:minimal, 
    gravity=gravity, 
    timestep=timestep, 
    damper=0.1, 
    spring=[1; 1.0/1000*scale*ones(num_u); scale/1000000*ones(num_links - num_u - 1)], #0.25e-323
    friction_coefficient=0.5,
    contact_feet=true, 
    contact_body=true,
    num_links=num_links,
    num_u=num_u,
    dim3=dim3,
    radius=radius,
    radius_b=radius_b,
    height=height,
    limits=false);

obs = reset(env)
env.state .= get_minimal_state(env.mechanism)
render(env)
# ## Open visualizer
body_position=[0.0; 0.0; radius + 0.1*radius]
stem_orientation = -40*π/180*0
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)
nu = num_u
render(env)
open(env.vis)

set_camera!(env.vis,
    cam_pos=[0.0, 0.0, 1.0],
    zoom=3.0)

# ## Inputs
if dim3
    num_u = env.num_inputs
    u_root = repeat([-0.0; 1],nu)*15/10000*scale
else
    u_root = 80/1000000*scale*[1*2.2;1.0] # [1; 1]*6/1000000*scale*2
end

body_position=[0.0; 0.0; radius + 0.1*radius]
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)


y = [copy(env.state)] # state trajectory
T = 101
ū = [zeros(env.num_inputs) for i in 1:T-1]
for t = 1:50
    ū[t] .= u_root/t
    step(env, env.state, ū[t])
    push!(y, copy(env.state)) 
end
for t = 51:100
    ū[t] .= -u_root*20/t
    step(env, env.state, ū[t])
    push!(y, copy(env.state)) 
end

visualize(env, y[1:end]);

storage = generate_storage(env.mechanism, [env.representation == :minimal ? minimal_to_maximal(env.mechanism, x) : x for x in y]);
res = get_sdf(env.mechanism, storage)


# TODO - Sweep magnitudes, fully actuated


# ---------------------Gait Generating Trajopt----------------------------------------

# ## dimensions
n = env.num_states
m = env.num_inputs


# ## model
dyn = IterativeLQR.Dynamics(
    (y, x, u, w) -> dynamics(y, env, x, u, w),
    (dx, x, u, w) -> dynamics_jacobian_state(dx, env, x, u, w),
    (du, x, u, w) -> dynamics_jacobian_input(du, env, x, u, w),
    n, n, m)

model = [dyn for t = 1:T-1]

# ## rollout
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
x1 = get_minimal_state(env.mechanism)
x̄ = IterativeLQR.rollout(model, x1, ū)
visualize(env, x̄);
timestep=0.05

xref = [[1.0*t/100; 0.0; radius; zeros(3); 0.1; zeros(5); zeros(4); repeat([0.0; 40*π/180*sin(2*π/(T-1)*t); 0.0; 0.0],2); zeros(4*(num_links-nu-1))] for t=0:T-1]
rel_c = 5000000
rel_c = 500
# ## objective
# qt = [1.0; 0.05; 1.0; 0.01 * ones(3); 0.5; 0.01 * ones(2); 0.01 * ones(3); fill([0.000000000001, 0.0000000000001], num_links-u*2)...]
qt = [100.0; 0.0; 100.0; 10.0; 10.0; 0.0; 10.0; zeros(5); zeros(4); repeat([10.0; 500.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))] 
qt_end = [100.0; 0.0; 100; 100.0*ones(3); 1.0; zeros(5); zeros(4); repeat([10.0; 500.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))]
# + 5*rel_c*timestep*((u[1]-u[3])^2 + (u[2]-u[4])^2)
ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [100000; 100000; 100000; 1.0]) * u + 5*rel_c*timestep*((u[1]-u[3])^2 + (u[2]-u[4])^2) for t = 1:T-1]
# ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [1.0; 1.0]) * u for t = 1:T-1]
oT = (x, u, w) -> transpose(x-xref[end]) * Diagonal(timestep * qt_end) * (x-xref[end])

cts = [IterativeLQR.Cost(ot, n, m) for ot in ots]
cT = IterativeLQR.Cost(oT, n, 0)
obj = [cts..., cT]

# ## constraints
function goal(x, u, w)
    Δ = x - xref
    return zeros(3)
end

function limits(x,u,w)
    [
        (0)*10;
        (0)*10;
    ]
end

cont = IterativeLQR.Constraint(limits,n,m,idx_ineq=collect(1:2m))
cont = IterativeLQR.Constraint()
# conT = IterativeLQR.Constraint(goal, n, 0)
cons = [cont for t = 1:T]

# ## solver
s = IterativeLQR.solver(model, obj, cons, 
    opts=IterativeLQR.Options(
        max_al_iter=10,
        obj_tol=1.0e-1,
        grad_tol=1.0e-2,
        verbose=true))

IterativeLQR.initialize_controls!(s, ū)
IterativeLQR.initialize_states!(s, x̄)

# ## solve
@time IterativeLQR.solve!(s)

# ## solution
x_sol, u_sol = IterativeLQR.get_trajectory(s)
@show IterativeLQR.eval_obj(s.m_data.obj.costs, s.m_data.x, s.m_data.u, s.m_data.w)
@show s.s_data.iter[1]
# @show norm(goal(s.m_data.x[T], zeros(0), zeros(0)), Inf)

# ## visualize
vis= Visualizer()
# open(env.vis)
x_view = [[x_sol[1] for t = 1:15]..., x_sol..., [x_sol[end] for t = 1:15]...]
visualize(env, x_sol);


#-------------------Optimal Gait with varying length---------------------------


name = "pleuro_5_link.jld2"
@load joinpath(@__DIR__, "results/"*name) x_sol u_sol

h_array = LinRange(0.1,0.25,30)
dist = []
for h in h_array
    num_links = 5
    num_u = 2
    dim3 = true
    timestep=0.05
    scale=1
    factor = 20/scale
    radius = 0.01*scale
    radius_b = 0.05*scale
    h_total = h
    height = h_total/num_links
    height_b = 2*radius
    density = 1500
    mass = π*radius^2*height*density
    mass_2 = mass
    mass_b = π*radius_b^2*height_b*density
    mass_total = (mass*num_u + mass_2*(num_links-num_u) + mass_b)
    Fg = 9.81*mass_total
    V_b = π*radius^2*height*num_links + π*radius_b^2*height_b
    Fb = V_b*1000*9.81
    # g = 9.81
    g = (Fg-Fb)/mass_total
    @show g
    gravity=[0.0; 0.0; -g]
    env = pleuro(
        representation=:minimal, 
        gravity=gravity, 
        timestep=timestep, 
        damper=0.1, 
        spring=[1; 1.0/1000*scale*ones(num_u); scale/1000000*ones(num_links - num_u - 1)], #0.25e-323
        friction_coefficient=0.5,
        contact_feet=true, 
        contact_body=true,
        num_links=num_links,
        num_u=num_u,
        dim3=dim3,
        radius=radius,
        radius_b=radius_b,
        height=height,
        limits=false);

    obs = reset(env)
    # initialize!(env.mechanism, Dojo.type2symbol(Patrick),
    #     body_position=[0.0, 0.0, 1.0], 
    #     body_orientation=[0.0, 0.0, 0.0],
    #     ankle_orientation=0.0)
    # initialize_patrick!(env.mechanism,
    #     body_position=[0.0, 0.0, 0.205], 
    #     body_orientation=[0.0, 0.0, 0.0],
    #     ankle_orientation=0.0)
    env.state .= get_minimal_state(env.mechanism)
    # render(env)
    # ## Open visualizer
    # initialize!(env.mechanism, Dojo.type2symbol(Patrick))
    body_position=[0.0; 0.0; radius + 0.1*radius]
    stem_orientation = -40*π/180*0
    initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
    env.state .= get_minimal_state(env.mechanism)
    # init_pos = [body_position;zeros(3);zeros(6);0.0;0*40*π/180;0.0;40*π/180;zeros(12)]
    # env.state = init_pos
    nu = num_u
    # render(env)
    y = [copy(env.state)] # state trajectory
    T = 101
    # open(env.vis)
for t = 1:T-1
        step(env, env.state, u_sol[t])
        push!(y, copy(env.state)) 
    end
    dist = [dist; y[end][1]]
    @show [h, y[end][1]]
end

plot(h_array,dist)
path = "echino-sim/trajopt/results/"
file = matopen(path*"dists_h_3"*".mat", "w")
write(file, "dists", dist)
close(file)

#-------------------Optimizing Gait with varying amplitude---------------------------

# ## dimensions
n = env.num_states
m = env.num_inputs


# ## model
dyn = IterativeLQR.Dynamics(
    (y, x, u, w) -> dynamics(y, env, x, u, w),
    (dx, x, u, w) -> dynamics_jacobian_state(dx, env, x, u, w),
    (du, x, u, w) -> dynamics_jacobian_input(du, env, x, u, w),
    n, n, m)

model = [dyn for t = 1:T-1]

# ## rollout
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
x1 = get_minimal_state(env.mechanism)
x̄ = IterativeLQR.rollout(model, x1, ū)
visualize(env, x̄);
timestep=0.05

xref = [[1.0*t/100; 0.0; radius; zeros(3); 0.1; zeros(5); zeros(4); repeat([0.0; 40*π/180*sin(2*π/(T-1)*t); 0.0; 0.0],2); zeros(4*(num_links-nu-1))] for t=0:T-1]
rel_c = 5000000
rel_c = 500
# ## objective
# qt = [1.0; 0.05; 1.0; 0.01 * ones(3); 0.5; 0.01 * ones(2); 0.01 * ones(3); fill([0.000000000001, 0.0000000000001], num_links-u*2)...]
qt = [100.0; 0.0; 100.0; 10.0; 10.0; 0.0; 10.0; zeros(5); zeros(4); repeat([10.0; 500.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))] 
qt_end = [100.0; 0.0; 100; 100.0*ones(3); 1.0; zeros(5); zeros(4); repeat([10.0; 500.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))]
# + 5*rel_c*timestep*((u[1]-u[3])^2 + (u[2]-u[4])^2)
ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [100000; 100000; 100000; 1.0]) * u + 5*rel_c*timestep*((u[1]-u[3])^2 + (u[2]-u[4])^2) for t = 1:T-1]
# ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [1.0; 1.0]) * u for t = 1:T-1]
oT = (x, u, w) -> transpose(x-xref[end]) * Diagonal(timestep * qt_end) * (x-xref[end])

cts = [IterativeLQR.Cost(ot, n, m) for ot in ots]
cT = IterativeLQR.Cost(oT, n, 0)
obj = [cts..., cT]

# ## constraints
function goal(x, u, w)
    Δ = x - xref
    return zeros(3)
end

function limits(x,u,w)
    [
        (ul-u)*10;
        (u-uu)*10;
    ]
end

cont = IterativeLQR.Constraint(limits,n,m,idx_ineq=collect(1:2m))
cont = IterativeLQR.Constraint()
# conT = IterativeLQR.Constraint(goal, n, 0)
cons = [cont for t = 1:T]

# ## solver
s = IterativeLQR.solver(model, obj, cons, 
    opts=IterativeLQR.Options(
        max_al_iter=10,
        obj_tol=1.0e-1,
        grad_tol=1.0e-2,
        verbose=true))

IterativeLQR.initialize_controls!(s, ū)
IterativeLQR.initialize_states!(s, x̄)

# ## solve
@time IterativeLQR.solve!(s)

# ## solution
x_sol, u_sol_new = IterativeLQR.get_trajectory(s)
@show IterativeLQR.eval_obj(s.m_data.obj.costs, s.m_data.x, s.m_data.u, s.m_data.w)
@show s.s_data.iter[1]
# @show norm(goal(s.m_data.x[T], zeros(0), zeros(0)), Inf)

# ## visualize
vis= Visualizer()
# open(env.vis)
x_view = [[x_sol[1] for t = 1:15]..., x_sol..., [x_sol[end] for t = 1:15]...]
visualize(env, x_sol);





#-------------------Optimal Gait with varying stiffness---------------------------


name = "pleuro_5_link.jld2"
@load joinpath(@__DIR__, "results/"*name) x_sol u_sol

s_array = LinRange(1000,10000000,10)
s_array = 10 .^ (range(-3,stop=-12,length=10))

dist_s = []
for s in s_array
    num_links = 5
    num_u = 2
    dim3 = true
    timestep=0.05
    scale=1
    factor = 20/scale
    radius = 0.01*scale
    radius_b = 0.05*scale
    h_total = 0.175
    height = h_total/num_links
    height_b = 2*radius
    density = 1500
    mass = π*radius^2*height*density
    mass_2 = mass
    mass_b = π*radius_b^2*height_b*density
    mass_total = (mass*num_u + mass_2*(num_links-num_u) + mass_b)
    Fg = 9.81*mass_total
    V_b = π*radius^2*height*num_links + π*radius_b^2*height_b
    Fb = V_b*1000*9.81
    # g = 9.81
    g = (Fg-Fb)/mass_total
    @show g
    gravity=[0.0; 0.0; -g]
    env = pleuro(
        representation=:minimal, 
        gravity=gravity, 
        timestep=timestep, 
        damper=0.1, 
        spring=[1; 1.0/1000*scale*ones(num_u); scale*s*ones(num_links - num_u - 1)], #0.25e-323
        friction_coefficient=0.5,
        contact_feet=true, 
        contact_body=true,
        num_links=num_links,
        num_u=num_u,
        dim3=dim3,
        radius=radius,
        radius_b=radius_b,
        height=height,
        limits=false);

    obs = reset(env)
    # initialize!(env.mechanism, Dojo.type2symbol(Patrick),
    #     body_position=[0.0, 0.0, 1.0], 
    #     body_orientation=[0.0, 0.0, 0.0],
    #     ankle_orientation=0.0)
    # initialize_patrick!(env.mechanism,
    #     body_position=[0.0, 0.0, 0.205], 
    #     body_orientation=[0.0, 0.0, 0.0],
    #     ankle_orientation=0.0)
    env.state .= get_minimal_state(env.mechanism)
    # render(env)
    # ## Open visualizer
    # initialize!(env.mechanism, Dojo.type2symbol(Patrick))
    body_position=[0.0; 0.0; radius + 0.1*radius]
    stem_orientation = -40*π/180*0
    initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
    env.state .= get_minimal_state(env.mechanism)
    # init_pos = [body_position;zeros(3);zeros(6);0.0;0*40*π/180;0.0;40*π/180;zeros(12)]
    # env.state = init_pos
    nu = num_u
    # render(env)
    y = [copy(env.state)] # state trajectory
    T = 101
    # open(env.vis)
    for t = 1:T-1
        step(env, env.state, u_sol[t])
        push!(y, copy(env.state)) 
    end
    dist_s = [dist_s; y[end][1]]
    @show [s, y[end][1]]
end

plot(s_array,dist_s)



#----------------------------------Optimizing Gait for various lengths------------------------------------------


name = "pleuro_5_link.jld2"
@load joinpath(@__DIR__, "results/"*name) x_sol u_sol

@load joinpath(@__DIR__, "results/pleuro_ilqr-new_body_0.175_2022-11-08 17:22:43.jld2") x_sol u_sol_new
u_sol = u_sol_new

h_array_opt = [0.1; 0.125; 0.15; 0.175; 0.2; 0.225]
num_links = 5
num_u = 2
dim3 = true
timestep=0.05
scale=1
factor = 20/scale
radius = 0.01*scale
radius_b = 0.05*scale
h_total = h_array_opt[1]
height = h_total/num_links
height_b = 2*radius
density = 1500
mass = π*radius^2*height*density
mass_2 = mass
mass_b = π*radius_b^2*height_b*density
mass_total = (mass*num_u + mass_2*(num_links-num_u) + mass_b)
Fg = 9.81*mass_total
V_b = π*radius^2*height*num_links + π*radius_b^2*height_b
Fb = V_b*1000*9.81
# g = 9.81
g = (Fg-Fb)/mass_total
@show g
gravity=[0.0; 0.0; -g]
env = pleuro(
    representation=:minimal, 
    gravity=gravity, 
    timestep=timestep, 
    damper=0.1, 
    spring=[1; 1.0/1000*scale*ones(num_u); scale/1000000*ones(num_links - num_u - 1)], #0.25e-323
    friction_coefficient=0.5,
    contact_feet=true, 
    contact_body=true,
    num_links=num_links,
    num_u=num_u,
    dim3=dim3,
    radius=radius,
    radius_b=radius_b,
    height=height,
    limits=false);

obs = reset(env)

env.state .= get_minimal_state(env.mechanism)
# render(env)
# ## Open visualizer
body_position=[0.0; 0.0; radius + 0.1*radius]
stem_orientation = -40*π/180*0
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)

nu = num_u
T = 101
# ## dimensions
n = env.num_states
m = env.num_inputs

# ## model
dyn = IterativeLQR.Dynamics(
    (y, x, u, w) -> dynamics(y, env, x, u, w),
    (dx, x, u, w) -> dynamics_jacobian_state(dx, env, x, u, w),
    (du, x, u, w) -> dynamics_jacobian_input(du, env, x, u, w),
    n, n, m)

model = [dyn for t = 1:T-1]

# ## rollout
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
x1 = get_minimal_state(env.mechanism)
ū = u_sol
x̄ = IterativeLQR.rollout(model, x1, ū)
visualize(env, x̄);
timestep=0.05

xref = [[1.0*t/100; 0.0; radius; zeros(3); 0.1; zeros(5); zeros(4); repeat([0.0; 40*π/180*sin(2*π/(T-1)*t); 0.0; 0.0],2); zeros(4*(num_links-nu-1))] for t=0:T-1]
rel_c = 5000000
rel_c = 500
# ## objective
qt = [100.0; 0.0; 100.0; 10.0; 10.0; 0.0; 10.0; zeros(5); zeros(4); repeat([0.0; 100.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))] 
qt_end = [100.0; 0.0; 100; 100.0*ones(3); 1.0; zeros(5); zeros(4); repeat([0.0; 100.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))]
ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [100000; 1.0; 100000; 1.0]) * u + 5*rel_c*timestep*((u[1]-u[3])^2 + (u[2]-u[4])^2) for t = 1:T-1]
oT = (x, u, w) -> transpose(x-xref[end]) * Diagonal(timestep * qt_end) * (x-xref[end])

cts = [IterativeLQR.Cost(ot, n, m) for ot in ots]
cT = IterativeLQR.Cost(oT, n, 0)
obj = [cts..., cT]

# ## constraints
function goal(x, u, w)
    Δ = x - xref
    return zeros(3)
end

function limits(x,u,w)
    [
        (0)*10;
        (0)*10;
    ]
end

cont = IterativeLQR.Constraint(limits,n,m,idx_ineq=collect(1:2m))
cont = IterativeLQR.Constraint()
# conT = IterativeLQR.Constraint(goal, n, 0)
cons = [cont for t = 1:T]

# ## solver
s = IterativeLQR.solver(model, obj, cons, 
    opts=IterativeLQR.Options(
        max_al_iter=10,
        obj_tol=1.0e-1,
        grad_tol=1.0e-2,
        verbose=true))

IterativeLQR.initialize_controls!(s, ū)
IterativeLQR.initialize_states!(s, x̄)

# ## solve
@time IterativeLQR.solve!(s)

# ## solution
x_sol, u_sol_new = IterativeLQR.get_trajectory(s)
@show IterativeLQR.eval_obj(s.m_data.obj.costs, s.m_data.x, s.m_data.u, s.m_data.w)
@show s.s_data.iter[1]
# @show norm(goal(s.m_data.x[T], zeros(0), zeros(0)), Inf)

# ## visualize
vis= Visualizer()
# open(env.vis)
x_view = [[x_sol[1] for t = 1:15]..., x_sol..., [x_sol[end] for t = 1:15]...]
visualize(env, x_sol);

@save joinpath(@__DIR__, "results/pleuro_ilqr-new_body_"*string(h_total)*"_"*Dates.format(now(), "yyyy-mm-dd HH:MM:SS")*".jld2") x_sol u_sol_new

open(env.vis)
#


#--------------------------- Other Direction --------------------------------------------------




h_array_opt = [0.15; 0.175; 0.2; 0.225]
num_links = 5
num_u = 2
dim3 = true
timestep=0.05
scale=1
factor = 20/scale
radius = 0.01*scale
radius_b = 0.05*scale
h_total = h_array_opt[1]
height = h_total/num_links
height_b = 2*radius
density = 1500
mass = π*radius^2*height*density
mass_2 = mass
mass_b = π*radius_b^2*height_b*density
mass_total = (mass*num_u + mass_2*(num_links-num_u) + mass_b)
Fg = 9.81*mass_total
V_b = π*radius^2*height*num_links + π*radius_b^2*height_b
Fb = V_b*1000*9.81
# g = 9.81
g = (Fg-Fb)/mass_total
@show g
gravity=[0.0; 0.0; -g]
env = pleuro(
    representation=:minimal, 
    gravity=gravity, 
    timestep=timestep, 
    damper=0.1, 
    spring=[1; 1.0/1000*scale*ones(num_u); scale/1000000*ones(num_links - num_u - 1)], #0.25e-323
    friction_coefficient=0.5,
    contact_feet=true, 
    contact_body=true,
    num_links=num_links,
    num_u=num_u,
    dim3=dim3,
    radius=radius,
    radius_b=radius_b,
    height=height,
    limits=false);

obs = reset(env)

env.state .= get_minimal_state(env.mechanism)
# render(env)
# ## Open visualizer
body_position=[0.0; 0.0; radius + 0.1*radius]
stem_orientation = -40*π/180*0
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)

nu = num_u
T = 101
# ## dimensions
n = env.num_states
m = env.num_inputs

# ## model
dyn = IterativeLQR.Dynamics(
    (y, x, u, w) -> dynamics(y, env, x, u, w),
    (dx, x, u, w) -> dynamics_jacobian_state(dx, env, x, u, w),
    (du, x, u, w) -> dynamics_jacobian_input(du, env, x, u, w),
    n, n, m)

model = [dyn for t = 1:T-1]

# ## rollout
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
x1 = get_minimal_state(env.mechanism)
ū = u_sol
x̄ = IterativeLQR.rollout(model, x1, ū)
visualize(env, x̄);
timestep=0.05

xref = [[-1.0*t/100; 0.0; radius; zeros(3); -0.1; zeros(5); zeros(4); repeat([0.0; 40*π/180*sin(2*π/(T-1)*t); 0.0; 0.0],2); zeros(4*(num_links-nu-1))] for t=0:T-1]
rel_c = 5000000
rel_c = 500
# ## objective
qt = [100.0; 0.0; 100.0; 10.0; 10.0; 0.0; 10.0; zeros(5); zeros(4); repeat([10.0; 500.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))] 
qt_end = [100.0; 0.0; 100; 100.0*ones(3); 1.0; zeros(5); zeros(4); repeat([10.0; 500.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))]
ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [100000; 100000; 100000; 1.0]) * u + 5*rel_c*timestep*((u[1]-u[3])^2 + (u[2]-u[4])^2) for t = 1:T-1]
oT = (x, u, w) -> transpose(x-xref[end]) * Diagonal(timestep * qt_end) * (x-xref[end])

cts = [IterativeLQR.Cost(ot, n, m) for ot in ots]
cT = IterativeLQR.Cost(oT, n, 0)
obj = [cts..., cT]

# ## constraints
function goal(x, u, w)
    Δ = x - xref
    return zeros(3)
end

function limits(x,u,w)
    [
        (0)*10;
        (0)*10;
    ]
end

cont = IterativeLQR.Constraint(limits,n,m,idx_ineq=collect(1:2m))
cont = IterativeLQR.Constraint()
# conT = IterativeLQR.Constraint(goal, n, 0)
cons = [cont for t = 1:T]

# ## solver
s = IterativeLQR.solver(model, obj, cons, 
    opts=IterativeLQR.Options(
        max_al_iter=10,
        obj_tol=1.0e-1,
        grad_tol=1.0e-2,
        verbose=true))

IterativeLQR.initialize_controls!(s, ū)
IterativeLQR.initialize_states!(s, x̄)

# ## solve
@time IterativeLQR.solve!(s)

# ## solution
x_sol, u_sol_new = IterativeLQR.get_trajectory(s)
@show IterativeLQR.eval_obj(s.m_data.obj.costs, s.m_data.x, s.m_data.u, s.m_data.w)
@show s.s_data.iter[1]
# @show norm(goal(s.m_data.x[T], zeros(0), zeros(0)), Inf)

# ## visualize
vis= Visualizer()
# open(env.vis)
x_view = [[x_sol[1] for t = 1:15]..., x_sol..., [x_sol[end] for t = 1:15]...]
visualize(env, x_sol);

@save joinpath(@__DIR__, "results/pleuro_ilqr-moving-other-way_"*Dates.format(now(), "yyyy-mm-dd HH:MM:SS")*".jld2") x_sol u_sol_new

open(env.vis)




#------------------------- Visualization ------------------------------------------------------------
name = "pleuro_ilqr-moving-other-way_2022-11-08 13:31:26.jld2" #"pleuro_5_link.jld2"
@load joinpath(@__DIR__, "results/"*name) x_sol u_sol_new
u_sol = u_sol_new

name = "pleuro_5_link.jld2"
@load joinpath(@__DIR__, "results/"*name) x_sol u_sol

@load joinpath(@__DIR__, "results/pleuro_ilqr-new_body_0.175_2022-11-08 17:22:43.jld2") x_sol u_sol_new
u_sol = u_sol_new

num_links = 5
num_u = 2
dim3 = true
timestep=0.05
scale=1
factor = 20/scale
radius = 0.01*scale
radius_b = 0.05*scale
h_total = 0.15*scale
height = h_total/num_links
height_b = 2*radius
density = 1500
mass = π*radius^2*height*density
mass_2 = mass
mass_b = π*radius_b^2*height_b*density
mass_total = (mass*num_u + mass_2*(num_links-num_u) + mass_b)
Fg = 9.81*mass_total
V_b = π*radius^2*height*num_links + π*radius_b^2*height_b
Fb = V_b*1000*9.81
# g = 9.81
g = (Fg-Fb)/mass_total
@show g
gravity=[0.0; 0.0; -g]
env = pleuro(
    representation=:minimal, 
    gravity=gravity, 
    timestep=timestep, 
    damper=0.1, 
    spring=[1; 1.0/1000*scale*ones(num_u); scale/1000000*ones(num_links - num_u - 1)], #0.25e-323
    friction_coefficient=0.5,
    contact_feet=true, 
    contact_body=true,
    num_links=num_links,
    num_u=num_u,
    dim3=dim3,
    radius=radius,
    radius_b=radius_b,
    height=height,
    limits=false);

obs = reset(env)
env.state .= get_minimal_state(env.mechanism)
render(env)
# ## Open visualizer
body_position=[0.0; 0.0; radius + 0.1*radius]
stem_orientation = -40*π/180*0
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)
nu = num_u
render(env)
open(env.vis)

set_camera!(env.vis,
cam_pos=[0.0, 0.0, 1.0],
zoom=3.0)


body_position=[0.0; 0.0; radius + 0.1*radius]
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)


y = [copy(env.state)] # state trajectory

ū = [u_sol[1:75];[diagm([1; -1; 1; -1])*u for u in u_sol[26:75]];u_sol[26:75]]
T = length(ū)-1
for t = 1:T-1
    step(env, env.state, ū[t])
    push!(y, copy(env.state)) 
end

visualize(env, y[1:end]);

time = T*timestep
avg_velocity = -y[end][1]/time*100
power = sum([600000*transpose(ū[t])* ū[t] for t=1:T-1])
COT = power/Fg/y[end][1]
@show avg_velocity
@show COT

#----------------------------  Trying new cost function ------------------------------------------------
name = "pleuro_5_link.jld2"
@load joinpath(@__DIR__, "results/"*name) x_sol u_sol
num_links = 5
num_u = 2
dim3 = true
timestep=0.05
scale=1
factor = 20/scale
radius = 0.01*scale
radius_b = 0.05*scale
h_total = 0.15*scale
height = h_total/num_links
height_b = 2*radius
density = 1500
mass = π*radius^2*height*density
mass_2 = mass
mass_b = π*radius_b^2*height_b*density
mass_total = (mass*num_u + mass_2*(num_links-num_u) + mass_b)
Fg = 9.81*mass_total
V_b = π*radius^2*height*num_links + π*radius_b^2*height_b
Fb = V_b*1000*9.81
# g = 9.81
g = (Fg-Fb)/mass_total
@show g
gravity=[0.0; 0.0; -g]
env = pleuro(
    representation=:minimal, 
    gravity=gravity, 
    timestep=timestep, 
    damper=0.1, 
    spring=[1; 1.0/1000*scale*ones(num_u); scale/1000000*ones(num_links - num_u - 1)], #0.25e-323
    friction_coefficient=0.5,
    contact_feet=true, 
    contact_body=true,
    num_links=num_links,
    num_u=num_u,
    dim3=dim3,
    radius=radius,
    radius_b=radius_b,
    height=height,
    limits=false);

obs = reset(env)
env.state .= get_minimal_state(env.mechanism)
render(env)
# ## Open visualizer
body_position=[0.0; 0.0; radius + 0.1*radius]
stem_orientation = -40*π/180*0
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)
nu = num_u
render(env)
open(env.vis)

# ## Inputs
if dim3
    num_u = env.num_inputs
    u_root = repeat([-0.0; 1],nu)*20/10000*scale
else
    u_root = 80/1000000*scale*[1*2.2;1.0] # [1; 1]*6/1000000*scale*2
end

body_position=[0.0; 0.0; radius + 0.1*radius]
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)


y = [copy(env.state)] # state trajectory
T = 101
ū = [zeros(env.num_inputs) for i in 1:T-1]
for t = 1:50
    ū[t] .= u_root/t
    step(env, env.state, ū[t])
    push!(y, copy(env.state)) 
end
for t = 51:100
    ū[t] .= -u_root*20/t
    step(env, env.state, ū[t])
    push!(y, copy(env.state)) 
end

visualize(env, y[1:end]);

storage = generate_storage(env.mechanism, [env.representation == :minimal ? minimal_to_maximal(env.mechanism, x) : x for x in y]);
res = get_sdf(env.mechanism, storage)


# ## dimensions
n = env.num_states
m = env.num_inputs


# ## model
dyn = IterativeLQR.Dynamics(
    (y, x, u, w) -> dynamics(y, env, x, u, w),
    (dx, x, u, w) -> dynamics_jacobian_state(dx, env, x, u, w),
    (du, x, u, w) -> dynamics_jacobian_input(du, env, x, u, w),
    n, n, m)

model = [dyn for t = 1:T-1]

# ## rollout
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
x1 = get_minimal_state(env.mechanism)
x̄ = IterativeLQR.rollout(model, x1, ū)
visualize(env, x̄);
timestep=0.05

xref = [[1.0*t/100; 0.0; radius; zeros(3); 0.1; zeros(5); zeros(4); repeat([0.0; 40*π/180*sin(2*π/(T-1)*t); 0.0; 0.0],2); zeros(4*(num_links-nu-1))] for t=0:T-1]
rel_c = 5000000
rel_c = 500
# ## objective
# qt = [1.0; 0.05; 1.0; 0.01 * ones(3); 0.5; 0.01 * ones(2); 0.01 * ones(3); fill([0.000000000001, 0.0000000000001], num_links-u*2)...]
qt = [100.0; 0.0; 100.0; 10.0; 10.0; 0.0; 10.0; zeros(5); zeros(4); repeat([0.0; 500.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))] 
qt_end = [100.0; 0.0; 100; 100.0*ones(3); 1.0; zeros(5); zeros(4); repeat([0.0; 500.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))]
# + 5*rel_c*timestep*((u[1]-u[3])^2 + (u[2]-u[4])^2)
ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [100000000; 1.0; 100000000; 1.0]) * u for t = 1:T-1]
# ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [1.0; 1.0]) * u for t = 1:T-1]
oT = (x, u, w) -> transpose(x-xref[end]) * Diagonal(timestep * qt_end) * (x-xref[end])

cts = [IterativeLQR.Cost(ot, n, m) for ot in ots]
cT = IterativeLQR.Cost(oT, n, 0)
obj = [cts..., cT]

# ## constraints
function goal(x, u, w)
    Δ = x - xref
    return zeros(3)
end

function limits(x,u,w)
    [
        (0)*10;
        (0)*10;
    ]
end

cont = IterativeLQR.Constraint(limits,n,m,idx_ineq=collect(1:2m))
cont = IterativeLQR.Constraint()
# conT = IterativeLQR.Constraint(goal, n, 0)
cons = [cont for t = 1:T]

# ## solver
s = IterativeLQR.solver(model, obj, cons, 
    opts=IterativeLQR.Options(
        max_al_iter=10,
        obj_tol=1.0e-1,
        grad_tol=1.0e-2,
        verbose=true))

IterativeLQR.initialize_controls!(s, ū)
IterativeLQR.initialize_states!(s, x̄)

# ## solve
@time IterativeLQR.solve!(s)

# ## solution
x_sol, u_sol = IterativeLQR.get_trajectory(s)
@show IterativeLQR.eval_obj(s.m_data.obj.costs, s.m_data.x, s.m_data.u, s.m_data.w)
@show s.s_data.iter[1]
# @show norm(goal(s.m_data.x[T], zeros(0), zeros(0)), Inf)

# ## visualize
vis= Visualizer()
# open(env.vis)
x_view = [[x_sol[1] for t = 1:15]..., x_sol..., [x_sol[end] for t = 1:15]...]
visualize(env, x_sol);


@save joinpath(@__DIR__, "results/pleuro_ilqr-reduced-y"*Dates.format(now(), "yyyy-mm-dd HH:MM:SS")*".jld2") x_sol u_sol










time = T*timestep
avg_velocity = -x_sol[end][1]/time*1000
power = sum([600000*transpose(u_sol[t])* u_sol[t] for t=1:T-1])
COT = power/Fg/-x_sol[end][1]
@show avg_velocity
@show COT

initialize_pleuro!(env.mechanism)
y = [copy(env.state)]
for i=1:3
    for t = 1:100
        if i==2
            neg = -1
        else
            neg = 1
        end
        step(env, env.state, neg*u_sol[t])
        push!(y, copy(env.state)) 
    end
end
visualize(env, y);

@save joinpath(@__DIR__, "results/pleuro_ilqr-"*Dates.format(now(), "yyyy-mm-dd HH:MM:SS")*".jld2") x_sol u_sol

name = "pleuro_5_link.jld2"
@load joinpath(@__DIR__, "results/"*name) x_sol u_sol


searchdir(path,key) = filter(x->contains(x,key), readdir(path))
path = "echino-sim/trajopt/results/using/segments/"
key = ".jld2"
for name in searchdir(path,key)
    @load joinpath(@__DIR__, "results/using/segments/"*name) x_sol u_sol
    @show x_sol[end][1]
    file = matopen(path*"x_"*name[1:end-5]*".mat", "w")
    write(file, "x_sol", x_sol)
    close(file)
    file = matopen(path*"u_"*name[1:end-5]*".mat", "w")
    write(file, "u_sol", u_sol)
    close(file)
end








# ----------------------------------- OG Optim ---------------------------------------

num_links = 5
num_u = 2
dim3 = true
timestep=0.05
scale=1
factor = 20/scale
radius = 0.01*scale
radius_b = 0.05*scale
h_total = 0.15*scale
height = h_total/num_links
height_b = 2*radius
density = 1500
mass = π*radius^2*height*density
mass_2 = mass
mass_b = π*radius_b^2*height_b*density
mass_total = (mass*num_u + mass_2*(num_links-num_u) + mass_b)
Fg = 9.81*mass_total
V_b = π*radius^2*height*num_links + π*radius_b^2*height_b
Fb = V_b*1000*9.81
# g = 9.81
g = (Fg-Fb)/mass_total
@show g
gravity=[0.0; 0.0; -g]
env = pleuro(
    representation=:minimal, 
    gravity=gravity, 
    timestep=timestep, 
    damper=0.1, 
    spring=[1; 1.0/1000*scale*ones(2); scale/1000000*ones(2)], #0.25e-323
    friction_coefficient=0.5,
    contact_feet=true, 
    contact_body=true,
    num_links=num_links,
    num_u=num_u,
    dim3=dim3,
    radius=radius,
    radius_b=radius_b,
    height=height,
    limits=false);

obs = reset(env)
# initialize!(env.mechanism, Dojo.type2symbol(Patrick),
#     body_position=[0.0, 0.0, 1.0], 
#     body_orientation=[0.0, 0.0, 0.0],
#     ankle_orientation=0.0)
# initialize_patrick!(env.mechanism,
#     body_position=[0.0, 0.0, 0.205], 
#     body_orientation=[0.0, 0.0, 0.0],
#     ankle_orientation=0.0)
env.state .= get_minimal_state(env.mechanism)
render(env)
# ## Open visualizer
# initialize!(env.mechanism, Dojo.type2symbol(Patrick))
body_position=[0.0; 0.0; radius + 0.1*radius]
stem_orientation = -40*π/180*0
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)
# init_pos = [body_position;zeros(3);zeros(6);0.0;0*40*π/180;0.0;40*π/180;zeros(12)]
# env.state = init_pos
render(env)
open(env.vis)

if dim3
    num_u = 4
    u_root = [-0.0; 1; -0.0; 1]*15/10000*scale
else
    u_root = 80/1000000*scale*[1*2.2;1.0] # [1; 1]*6/1000000*scale*2
end

body_position=[0.0; 0.0; radius + 0.1*radius]
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)
# env.state = init_pos

# randn(env.num_inputs)
y = [copy(env.state)] # state trajectory
T = 101
ū = [zeros(env.num_inputs) for i in 1:T-1]
for t = 1:50
    # ū[t] .= 5/100000*scale*ones(num_u)*sin(t/100*2*π)
    ū[t] .= u_root/t
    # ū[t] .= [-0.0; 1; -0.0; 1]*6/10000*scale*2
    # ū[t] .= 6/100000*scale*ones(num_u)*10
    # ū[t][1] = 4/100000*scale*sin(t/80*2*π)
    # ū[t][2:3] = -2/100000*scale*ones(num_links-1)*sin(t/80*2*π)
    # ū[t][1] = 1/200000*scale*sin(t/40*2*π)
    # ū[t][2] = 1/100000*scale*sin(t/40*2*π+1π/3)
    #ū[t][3] = 1/100000*scale*sin(t/40*2*π+2π/3)
    #ū[t] = 0.000006*ones(num_links)*0.0
    step(env, env.state, ū[t])
    push!(y, copy(env.state)) 
end
for t = 51:100
    ū[t] .= -u_root*20/t
    # ū[t] .= -6/100000*scale*ones(num_u)*10
    # ū[t][1] = 4/100000*scale*sin(t/80*2*π)
    # ū[t][2:3] = -2/100000*scale*ones(num_links-1)*sin(t/80*2*π)
    # ū[t][1] = 4/100000*scale*sin(t/80*2*π)
    # ū[t][2:3] = -2/100000*scale*ones(num_links-1)*sin(t/80*2*π)
    #ū[t] = 0.000006*ones(num_links)*0.0
    step(env, env.state, ū[t])
    push!(y, copy(env.state)) 
end

ul = -1/10000*scale*ones(num_u)
uu = 1/10000*scale*ones(num_u)
# for t = 1:100   
#     ū[t] = 2/100*scale*ones(num_links)
#     step(env, env.state, ū[t])
#     push!(y, copy(env.state)) 
# end
# sinusoidal
# ū[t][1] = 0.2/3*sin(t/80*2*π)
# ū[t][2:3] = -0.3/3*ones(2)*sin(t/80*2*π)

visualize(env, y[1:end]);

storage = generate_storage(env.mechanism, [env.representation == :minimal ? minimal_to_maximal(env.mechanism, x) : x for x in y]);
res = get_sdf(env.mechanism, storage)







# ## dimensions
n = env.num_states
m = env.num_inputs

# ## reference trajectory
#N = 2
#visualize(env, xref)

# storage = simulate!(mech, 1.0,
#     record=true,
#     verbose=false)

# visualize(mech, storage,
#     vis=env.vis)
# ## horizon


# ## model
dyn = IterativeLQR.Dynamics(
    (y, x, u, w) -> dynamics(y, env, x, u, w),
    (dx, x, u, w) -> dynamics_jacobian_state(dx, env, x, u, w),
    (du, x, u, w) -> dynamics_jacobian_input(du, env, x, u, w),
    n, n, m)

model = [dyn for t = 1:T-1]

# ## rollout
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
x1 = get_minimal_state(env.mechanism)
x̄ = IterativeLQR.rollout(model, x1, ū)
visualize(env, x̄);
# xgoal = [-0.8; 0.0; 0.2; zeros(3); -0.5; zeros(5); 0.7; -0.52; -0.3; -0.5; 0.08; 0.04; 0.02; 0.002]
# xgoal = [-1.0; 0.0; radius; zeros(3); -0.5; zeros(5); 0.5235997717304138;
# 4.632832888429317e-6;
# -0.1831888197137249;
# -0.49900709106006313;
# -0.2888118773615937;
# -0.5164038499880537;
# 0.16204649911700442;
# 0.06905133847779053;
# 0.08347651824670131;
# 0.02591559035806034;
# 0.026400856491096723;
# 0.0016440154564128688]
timestep=0.05

xref = [[1.0*t/100; 0.0; radius; zeros(3); 0.1; zeros(5); zeros(4); repeat([0.0; 40*π/180*sin(2*π/(T-1)*t); 0.0; 0.0],2); zeros(8)] for t=0:T-1]
rel_c = 5000000
rel_c = 500
# ## objective
# qt = [1.0; 0.05; 1.0; 0.01 * ones(3); 0.5; 0.01 * ones(2); 0.01 * ones(3); fill([0.000000000001, 0.0000000000001], num_links-u*2)...]
qt = [100.0; 0.0; 100.0; 10.0; 10.0; 0.0; 10.0; zeros(5); zeros(4); repeat([0.0; 500.0; 0.0; 0.0],2); zeros(8)] 
qt_end = [100.0; 0.0; 100; 100.0*ones(3); 1.0; zeros(5); zeros(4); repeat([0.0; 500.0; 0.0; 0.0],2); zeros(8)]
# + 5*rel_c*timestep*((u[1]-u[3])^2 + (u[2]-u[4])^2)
ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [100000; 1.0; 100000; 1.0]) * u + 5*rel_c*timestep*((u[1]-u[3])^2 + (u[2]-u[4])^2) for t = 1:T-1]
# ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [1.0; 1.0]) * u for t = 1:T-1]
oT = (x, u, w) -> transpose(x-xref[end]) * Diagonal(timestep * qt_end) * (x-xref[end])

cts = [IterativeLQR.Cost(ot, n, m) for ot in ots]
cT = IterativeLQR.Cost(oT, n, 0)
obj = [cts..., cT]

# ## constraints
function goal(x, u, w)
    Δ = x - xref
    return zeros(3)
end

function limits(x,u,w)
    [
        (ul-u)*10;
        (u-uu)*10;
    ]
end

cont = IterativeLQR.Constraint(limits,n,m,idx_ineq=collect(1:2m))
cont = IterativeLQR.Constraint()
# conT = IterativeLQR.Constraint(goal, n, 0)
cons = [cont for t = 1:T]

# ## solver
s = IterativeLQR.solver(model, obj, cons, 
    opts=IterativeLQR.Options(
        max_al_iter=10,
        obj_tol=1.0e-1,
        grad_tol=1.0e-2,
        verbose=true))
# s = IterativeLQR.solver(model, obj, cons,
#     opts=IterativeLQR.Options(
#         verbose=true,
#         linesearch=:armijo,
#         α_min=1.0e-5,
#         obj_tol=1.0e-3,
#         grad_tol=1.0e-3,
#         max_iter=100,
#         max_al_iter=5,
#         ρ_init=1.0,
#         ρ_scale=10.0))
IterativeLQR.initialize_controls!(s, ū)
IterativeLQR.initialize_states!(s, x̄)

# ## solve
@time IterativeLQR.solve!(s)

# ## solution
x_sol, u_sol = IterativeLQR.get_trajectory(s)
@show IterativeLQR.eval_obj(s.m_data.obj.costs, s.m_data.x, s.m_data.u, s.m_data.w)
@show s.s_data.iter[1]
# @show norm(goal(s.m_data.x[T], zeros(0), zeros(0)), Inf)

# ## visualize
vis= Visualizer()
# open(env.vis)
x_view = [[x_sol[1] for t = 1:15]..., x_sol..., [x_sol[end] for t = 1:15]...]
visualize(env, x_sol);








#----------------------------------New body plan Optimized Gait------------------------------------------


name = "pleuro_5_link.jld2"
@load joinpath(@__DIR__, "results/"*name) x_sol u_sol

h_array_opt = [0.15; 0.175; 0.2; 0.225]
num_links = 5
num_u = 2
dim3 = true
timestep=0.05
scale=1
factor = 20/scale
radius = 0.01*scale
radius_b = 0.05*scale
h_total = h_array_opt[4]
height = h_total/num_links
height_b = 2*radius
density = 1500
mass = π*radius^2*height*density
mass_2 = mass
mass_b = π*radius_b^2*height_b*density
mass_total = (mass*num_u + mass_2*(num_links-num_u) + mass_b)
Fg = 9.81*mass_total
V_b = π*radius^2*height*num_links + π*radius_b^2*height_b
Fb = V_b*1000*9.81
# g = 9.81
g = (Fg-Fb)/mass_total
@show g
gravity=[0.0; 0.0; -g]
env = pleuro(
    representation=:minimal, 
    gravity=gravity, 
    timestep=timestep, 
    damper=0.1, 
    spring=[1; 1.0/1000*scale*ones(num_u); scale/1000000*ones(num_links - num_u - 1)], #0.25e-323
    friction_coefficient=0.5,
    contact_feet=true, 
    contact_body=true,
    num_links=num_links,
    num_u=num_u,
    dim3=dim3,
    radius=radius,
    radius_b=radius_b,
    height=height,
    limits=false);

obs = reset(env)

env.state .= get_minimal_state(env.mechanism)
# render(env)
# ## Open visualizer
body_position=[0.0; 0.0; radius + 0.1*radius]
stem_orientation = -40*π/180*0
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)

nu = num_u
T = 101
# ## dimensions
n = env.num_states
m = env.num_inputs

# ## model
dyn = IterativeLQR.Dynamics(
    (y, x, u, w) -> dynamics(y, env, x, u, w),
    (dx, x, u, w) -> dynamics_jacobian_state(dx, env, x, u, w),
    (du, x, u, w) -> dynamics_jacobian_input(du, env, x, u, w),
    n, n, m)

model = [dyn for t = 1:T-1]

# ## rollout
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
x1 = get_minimal_state(env.mechanism)
ū = u_sol
x̄ = IterativeLQR.rollout(model, x1, ū)
visualize(env, x̄);
timestep=0.05

xref = [[1.0*t/100; 0.0; radius; zeros(3); 0.1; zeros(5); zeros(4); repeat([0.0; 40*π/180*sin(2*π/(T-1)*t); 0.0; 0.0],2); zeros(4*(num_links-nu-1))] for t=0:T-1]
rel_c = 5000000
rel_c = 500
# ## objective
qt = [100.0; 0.0; 100.0; 10.0; 10.0; 0.0; 10.0; zeros(5); zeros(4); repeat([0.0; 100.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))] 
qt_end = [100.0; 0.0; 100; 100.0*ones(3); 1.0; zeros(5); zeros(4); repeat([0.0; 100.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))]
ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [100000; 1.0; 100000; 1.0]) * u + 5*rel_c*timestep*((u[1]-u[3])^2 + (u[2]-u[4])^2) for t = 1:T-1]
oT = (x, u, w) -> transpose(x-xref[end]) * Diagonal(timestep * qt_end) * (x-xref[end])

cts = [IterativeLQR.Cost(ot, n, m) for ot in ots]
cT = IterativeLQR.Cost(oT, n, 0)
obj = [cts..., cT]

# ## constraints
function goal(x, u, w)
    Δ = x - xref
    return zeros(3)
end

function limits(x,u,w)
    [
        (0)*10;
        (0)*10;
    ]
end

cont = IterativeLQR.Constraint(limits,n,m,idx_ineq=collect(1:2m))
cont = IterativeLQR.Constraint()
# conT = IterativeLQR.Constraint(goal, n, 0)
cons = [cont for t = 1:T]

# ## solver
s = IterativeLQR.solver(model, obj, cons, 
    opts=IterativeLQR.Options(
        max_al_iter=10,
        obj_tol=1.0e-1,
        grad_tol=1.0e-2,
        verbose=true))

IterativeLQR.initialize_controls!(s, ū)
IterativeLQR.initialize_states!(s, x̄)

# ## solve
@time IterativeLQR.solve!(s)

# ## solution
x_sol, u_sol_new = IterativeLQR.get_trajectory(s)
@show IterativeLQR.eval_obj(s.m_data.obj.costs, s.m_data.x, s.m_data.u, s.m_data.w)
@show s.s_data.iter[1]
# @show norm(goal(s.m_data.x[T], zeros(0), zeros(0)), Inf)

# ## visualize
vis= Visualizer()
# open(env.vis)
x_view = [[x_sol[1] for t = 1:15]..., x_sol..., [x_sol[end] for t = 1:15]...]
visualize(env, x_sol);

@save joinpath(@__DIR__, "results/pleuro_ilqr-"*string(h_total)*"_"*Dates.format(now(), "yyyy-mm-dd HH:MM:SS")*".jld2") x_sol u_sol_new

open(env.vis)
#


#----------------------------------Fully actuated ------------------------------------------


#include(joinpath(@__DIR__, "algorithms/ags.jl")) # augmented random search
include(joinpath(@__DIR__, "../environments/pleuro/methods/initialize.jl"))
include(joinpath(@__DIR__, "../environments/pleuro/methods/env.jl"))
# ## Ant
num_links = 5
num_u = 4
dim3 = true
timestep=0.05
scale=1
factor = 20/scale
radius = 0.01*scale
radius_b = 0.05*scale
h_total = 0.175*scale
height = h_total/num_links
height_b = 2*radius
density = 1500
mass = π*radius^2*height*density
mass_2 = mass
mass_b = π*radius_b^2*height_b*density
mass_total = (mass*num_u + mass_2*(num_links-num_u) + mass_b)
Fg = 9.81*mass_total
V_b = π*radius^2*height*num_links + π*radius_b^2*height_b
Fb = V_b*1000*9.81
# g = 9.81
g = (Fg-Fb)/mass_total
@show g
gravity=[0.0; 0.0; -g]
env = pleuro(
    representation=:minimal, 
    gravity=gravity, 
    timestep=timestep, 
    damper=0.1, 
    spring=[1; 1.0/1000*scale*ones(num_u)], #0.25e-323
    friction_coefficient=0.5,
    contact_feet=true, 
    contact_body=true,
    num_links=num_links,
    num_u=num_u,
    dim3=dim3,
    radius=radius,
    radius_b=radius_b,
    height=height,
    limits=false);

obs = reset(env)
env.state .= get_minimal_state(env.mechanism)
render(env)
# ## Open visualizer
body_position=[0.0; 0.0; radius + 0.1*radius]
stem_orientation = -40*π/180*0
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)
nu = num_u
render(env)
open(env.vis)

set_camera!(env.vis,
    cam_pos=[0.0, 0.0, 1.0],
    zoom=3.0)

@load joinpath(@__DIR__, "results/pleuro_ilqr-new_body_0.175_2022-11-08 17:22:43.jld2") x_sol u_sol_new

initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)
# init_pos = [body_position;zeros(3);zeros(6);0.0;0*40*π/180;0.0;40*π/180;zeros(12)]
# env.state = init_pos
nu = num_u
# render(env)
y = [copy(env.state)] # state trajectory
T = 101
# open(env.vis)
# u_nom = [[u_sol_new[t]; -Diagonal([-1;-1;-1;-1])*u_sol_new[t]] for t=1:T-1]
u_nom = [[repeat([-0.0; 1],2)*15/10000/t*scale;repeat([-0.0; 1],2)*15/10000/t*scale] for t=1:T-1]
for t = 1:50
    
    step(env, env.state, u_nom[t])
    push!(y, copy(env.state)) 
end
for t = 51:T-1
    u_nom[t] = -u_nom[t]*10#*20
    step(env, env.state, u_nom[t])
    push!(y, copy(env.state)) 
end

open(env.vis)
visualize(env, y);

nu = num_u
T = 101
# ## dimensions
n = env.num_states
m = env.num_inputs

# ## model
dyn = IterativeLQR.Dynamics(
    (y, x, u, w) -> dynamics(y, env, x, u, w),
    (dx, x, u, w) -> dynamics_jacobian_state(dx, env, x, u, w),
    (du, x, u, w) -> dynamics_jacobian_input(du, env, x, u, w),
    n, n, m)

model = [dyn for t = 1:T-1]

# ## rollout
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
x1 = get_minimal_state(env.mechanism)
ū = u_nom
x̄ = IterativeLQR.rollout(model, x1, ū)
visualize(env, x̄);
timestep=0.05

xref = [[-1.0*t/100; 0.0; radius; zeros(3); -0.1; zeros(5); zeros(4); repeat([0.0; 40*π/180*sin(2*π/(T-1)*t); 0.0; 0.0],2); repeat([0.0; 40*π/180*sin(2*π/(T-1)*t); 0.0; 0.0],2)] for t=0:T-1]
rel_c = 5000000
rel_c = 500
# ## objective
qt = [100.0; 0.0; 100.0; 10.0; 10.0; 0.0; 10.0; zeros(5); zeros(4); repeat([0.0; 100.0; 0.0; 0.0],4)] 
qt_end = [100.0; 0.0; 100; 100.0*ones(3); 1.0; zeros(5); zeros(4); repeat([0.0; 100.0; 0.0; 0.0],4)]
ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [100000; 1.0; 100000; 1.0; 100000; 0.5; 100000; 0.25]) * u + 5*rel_c*timestep*((u[1]-u[3])^2 + (u[2]-u[4])^2 + (u[5]-u[7])^2 + (u[6]-u[8])^2)  for t = 1:T-1]
oT = (x, u, w) -> transpose(x-xref[end]) * Diagonal(timestep * qt_end) * (x-xref[end])

cts = [IterativeLQR.Cost(ot, n, m) for ot in ots]
cT = IterativeLQR.Cost(oT, n, 0)
obj = [cts..., cT]

# ## constraints
function goal(x, u, w)
    Δ = x - xref
    return zeros(3)
end

function limits(x,u,w)
    [
        (0)*10;
        (0)*10;
    ]
end

cont = IterativeLQR.Constraint(limits,n,m,idx_ineq=collect(1:2m))
cont = IterativeLQR.Constraint()
# conT = IterativeLQR.Constraint(goal, n, 0)
cons = [cont for t = 1:T]

# ## solver
s = IterativeLQR.solver(model, obj, cons, 
    opts=IterativeLQR.Options(
        max_al_iter=10,
        obj_tol=1.0e-1,
        grad_tol=1.0e-2,
        verbose=true))

IterativeLQR.initialize_controls!(s, ū)
IterativeLQR.initialize_states!(s, x̄)

# ## solve
@time IterativeLQR.solve!(s)

# ## solution
x_sol, u_sol_new = IterativeLQR.get_trajectory(s)
@show IterativeLQR.eval_obj(s.m_data.obj.costs, s.m_data.x, s.m_data.u, s.m_data.w)
@show s.s_data.iter[1]
# @show norm(goal(s.m_data.x[T], zeros(0), zeros(0)), Inf)

# ## visualize
vis= Visualizer()
# open(env.vis)
x_view = [[x_sol[1] for t = 1:15]..., x_sol..., [x_sol[end] for t = 1:15]...]
open(env.vis)
visualize(env, x_sol);

@save joinpath(@__DIR__, "results/pleuro_ilqr-full-opposite-"*string(h_total)*"_"*Dates.format(now(), "yyyy-mm-dd HH:MM:SS")*".jld2") x_sol u_sol_new

name = "pleuro_ilqr-sin-opposite-0.175_2022-11-21 19:52:04"
name = "pleuro_ilqr-full-opposite-0.175_2022-11-21 21:06:24"
name = "pleuro_ilqr-sin-0.175_2022-11-21 20:36:27"
name = "pleuro_ilqr-full-0.175_2022-11-21 19:24:47"
name = "pleuro_ilqr-moving-other-way_2022-11-08 13:31:26"
name = "pleuro_ilqr-new_body_0.175_2022-11-08 17:22:43"
name = "pleuro_ilqr-sin-0.175_2022-11-16 11:58:39"
@load joinpath(@__DIR__, "results/"*name*".jld2") x_sol u_sol_new

path = "echino-sim/trajopt/results/"
file = matopen(path*"half"*".mat", "w")
write(file, "x", x_sol)
write(file, "u", u_sol_new)

close(file)

time = T*timestep
avg_velocity = x_sol[end][1]/time*1000
power = sum([600000*transpose(u_sol_new[t])* u_sol_new[t] for t=1:T-1])
COT = power/Fg/x_sol[end][1]
@show avg_velocity
@show COT



#----------------------------------Bigger Sweep ------------------------------------------



#include(joinpath(@__DIR__, "algorithms/ags.jl")) # augmented random search
include(joinpath(@__DIR__, "../environments/pleuro/methods/initialize.jl"))
include(joinpath(@__DIR__, "../environments/pleuro/methods/env.jl"))
# ## Ant
num_links = 5
num_u = 2
dim3 = true
timestep=0.05
scale=1
factor = 20/scale
radius = 0.01*scale
radius_b = 0.05*scale
h_total = 0.15*scale
height = h_total/num_links
height_b = 2*radius
density = 1500
mass = π*radius^2*height*density
mass_2 = mass
mass_b = π*radius_b^2*height_b*density
mass_total = (mass*num_u + mass_2*(num_links-num_u) + mass_b)
Fg = 9.81*mass_total
V_b = π*radius^2*height*num_links + π*radius_b^2*height_b
Fb = V_b*1000*9.81
# g = 9.81
g = (Fg-Fb)/mass_total
@show g
gravity=[0.0; 0.0; -g]
env = pleuro(
    representation=:minimal, 
    gravity=gravity, 
    timestep=timestep, 
    damper=0.1, 
    spring=[1; 1.0/1000*scale*ones(num_u); scale/1000000*ones(num_links - num_u - 1)], #0.25e-323
    friction_coefficient=0.5,
    contact_feet=true, 
    contact_body=true,
    num_links=num_links,
    num_u=num_u,
    dim3=dim3,
    radius=radius,
    radius_b=radius_b,
    height=height,
    limits=false);

obs = reset(env)
env.state .= get_minimal_state(env.mechanism)
render(env)
# ## Open visualizer
body_position=[0.0; 0.0; radius + 0.1*radius]
stem_orientation = -40*π/180*0
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)
nu = num_u
render(env)
open(env.vis)

set_camera!(env.vis,
    cam_pos=[0.0, 0.0, 1.0],
    zoom=3.0)

# ## Inputs
if dim3
    num_u = env.num_inputs
    u_root = repeat([-0.0; 1],nu)*15/10000*scale
else
    u_root = 80/1000000*scale*[1*2.2;1.0] # [1; 1]*6/1000000*scale*2
end

body_position=[0.0; 0.0; radius + 0.1*radius]
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
env.state .= get_minimal_state(env.mechanism)


y = [copy(env.state)] # state trajectory
T = 101
ū = [zeros(env.num_inputs) for i in 1:T-1]
for t = 1:50
    ū[t] .= u_root/t
    step(env, env.state, ū[t])
    push!(y, copy(env.state)) 
end
for t = 51:100
    ū[t] .= -u_root*20/t
    step(env, env.state, ū[t])
    push!(y, copy(env.state)) 
end

visualize(env, y[1:end]);

storage = generate_storage(env.mechanism, [env.representation == :minimal ? minimal_to_maximal(env.mechanism, x) : x for x in y]);
res = get_sdf(env.mechanism, storage)


# TODO - Sweep magnitudes, fully actuated


# ---------------------Gait Generating Trajopt----------------------------------------

# ## dimensions
n = env.num_states
m = env.num_inputs


# ## model
dyn = IterativeLQR.Dynamics(
    (y, x, u, w) -> dynamics(y, env, x, u, w),
    (dx, x, u, w) -> dynamics_jacobian_state(dx, env, x, u, w),
    (du, x, u, w) -> dynamics_jacobian_input(du, env, x, u, w),
    n, n, m)

model = [dyn for t = 1:T-1]

# ## rollout
initialize_pleuro!(env.mechanism,body_position=body_position,stem_orientation=stem_orientation)
x1 = get_minimal_state(env.mechanism)
ū = u_sol_new
x̄ = IterativeLQR.rollout(model, x1, ū)
visualize(env, x̄);
timestep=0.05

xref = [[1.0*t/100; 0.0; radius; zeros(3); 0.1; zeros(5); zeros(4); repeat([0.0; 60*π/180*sin(2*π/(T-1)*t); 0.0; 0.0],2); zeros(4*(num_links-nu-1))] for t=0:T-1]
rel_c = 5000000
rel_c = 500
# ## objective
# qt = [1.0; 0.05; 1.0; 0.01 * ones(3); 0.5; 0.01 * ones(2); 0.01 * ones(3); fill([0.000000000001, 0.0000000000001], num_links-u*2)...]
qt = [100.0; 0.0; 100.0; 10.0; 10.0; 0.0; 10.0; zeros(5); zeros(4); repeat([10.0; 500.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))] 
qt_end = [500.0; 0.0; 100; 100.0*ones(3); 1.0; zeros(5); zeros(4); repeat([10.0; 500.0; 0.0; 0.0],2); zeros(4*(num_links-nu-1))]
# + 5*rel_c*timestep*((u[1]-u[3])^2 + (u[2]-u[4])^2)
ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [100000; 100000; 100000; 1.0]) * u + 5*rel_c*timestep*((u[1]-u[3])^2 + (u[2]-u[4])^2) for t = 1:T-1]
# ots = [(x, u, w) -> transpose(x-xref[t]) * Diagonal(timestep * qt) * (x-xref[t]) + rel_c*transpose(u) * Diagonal(timestep * [1.0; 1.0]) * u for t = 1:T-1]
oT = (x, u, w) -> transpose(x-xref[end]) * Diagonal(timestep * qt_end) * (x-xref[end])

cts = [IterativeLQR.Cost(ot, n, m) for ot in ots]
cT = IterativeLQR.Cost(oT, n, 0)
obj = [cts..., cT]

# ## constraints
function goal(x, u, w)
    Δ = x - xref
    return zeros(3)
end

function limits(x,u,w)
    [
        (0)*10;
        (0)*10;
    ]
end

cont = IterativeLQR.Constraint(limits,n,m,idx_ineq=collect(1:2m))
cont = IterativeLQR.Constraint()
# conT = IterativeLQR.Constraint(goal, n, 0)
cons = [cont for t = 1:T]

# ## solver
s = IterativeLQR.solver(model, obj, cons, 
    opts=IterativeLQR.Options(
        max_al_iter=10,
        obj_tol=1.0e-1,
        grad_tol=1.0e-2,
        verbose=true))

IterativeLQR.initialize_controls!(s, ū)
IterativeLQR.initialize_states!(s, x̄)

# ## solve
@time IterativeLQR.solve!(s)

# ## solution
x_sol, u_sol = IterativeLQR.get_trajectory(s)
@show IterativeLQR.eval_obj(s.m_data.obj.costs, s.m_data.x, s.m_data.u, s.m_data.w)
@show s.s_data.iter[1]
# @show norm(goal(s.m_data.x[T], zeros(0), zeros(0)), Inf)

# ## visualize
vis= Visualizer()
# open(env.vis)
x_view = [[x_sol[1] for t = 1:15]..., x_sol..., [x_sol[end] for t = 1:15]...]
visualize(env, x_sol);

@save joinpath(@__DIR__, "results/pleuro_ilqr-20deg-"*string(h_total)*"_"*Dates.format(now(), "yyyy-mm-dd HH:MM:SS")*".jld2") x_sol u_sol_new


path = "echino-sim/trajopt/results/"
file = matopen(path*"half-opp"*".mat", "w")
write(file, "x", x_sol)
close(file)

time = T*timestep
avg_velocity = x_sol[end][1]/time*1000
power = sum([600000*transpose(u_sol_new[t])* u_sol_new[t] for t=1:T-1])
COT = power/Fg/x_sol[end][1]
@show avg_velocity
@show COT