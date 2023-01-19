function get_pleuro(; 
    timestep=0.05, 
    gravity=[0.0; 0.0; -9.81], 
    friction_coefficient=0.5,
    spring=0.0, 
    damper=1.0, 
    contact_feet=true, 
    contact_body=true,
    contact_type=:nonlinear,
    num_links=2,
    num_u=1,
    dim3=false,
    limits::Bool=true,
    joint_limits=[-45*ones(num_links) * π / 180.0,
                    45*ones(num_links) * π / 180.0],
    radius=0.2/20,
    radius_b=1/20,
    height=0.5/20,
    T=Float64)

    # define origin
    origin = Origin{T}(name=:origin)
    # define the two bodies which will form the tippe top once assembled
    height_1 = height
    #height_2 = height
    height_b = 2*radius
    mass = π*radius^2*height_1*1500
    mass_2 = mass/2
    mass_b = π*radius_b^2*height_b*1500
    α = 0.2
    purple = RGBA(165.0/255.0, 55.0/255.0, 253.0/255.0, 1.0)
    bodies = [Cylinder(radius_b, height_b, mass_b, name=:body, color=Dojo.gray_light); Cylinder(radius, height_1/100, mass, color=purple); [Cylinder(radius, height_1, mass, color=purple) for i = 1:num_links-1]]
    
    bodies[1].inertia = 1 / 2 * mass_b * Diagonal([radius_b^2 + 1 / 6 * height_b^2; radius_b^2 + 1 / 6 * height_b^2; radius_b^2])
    # bodies = [
    #     Cylinder(radius_b, height_b, 4.0*mass, name=:body, color=Dojo.gray_light),
    #     Cylinder(radius, height_1, mass, name=:arm11, color=Dojo.gray_light),
    #     Cylinder(radius, height_1, mass, name=:arm12, color=Dojo.gray_light),
    #     Cylinder(radius, height_2, mass_2, name=:arm13, color=Dojo.gray_light),
    #     Cylinder(radius, height_2, mass_2, name=:arm14, color=Dojo.gray_light)]
    # modify inertia of the first body to ensure we obtain the desired behavior
    #bodies[1].inertia = Diagonal([1.9, 2.1, 2.0])

    # define the joints, the 1st joint is between Origin and :sphere1
    # the 2nd joint is between :sphere1 and :sphere2
    if dim3
        joints = [
        JointConstraint(Floating(origin, bodies[1]),
                name=:floating_joint);
            JointConstraint(Orbital(bodies[1], bodies[2],[0.0,0.0,1.0],
                axis_offset=Dojo.RotY(π/2-0.1),
                parent_vertex=[-radius_b,0,0],
                child_vertex=[0,0,height_1/2/10]),
                name = :joint1); 
            JointConstraint(Orbital(bodies[2], bodies[3],[0.0,0.0,1.0],
                parent_vertex=[0,0,-height_1/2/100],
                child_vertex=[0,0,height_1/2]),
                name = :joint2); 
            [JointConstraint(Orbital(bodies[i], bodies[i+1],[0.0,0.0,1.0],
                parent_vertex=[0,0,-height_1/2],
                child_vertex=[0,0,height_1/2]),
                name = Symbol("joint",i)) for i = 3:num_links]
        ]
    else
        joints = [
            JointConstraint(Floating(origin, bodies[1]),
                    name=:floating_joint);
            JointConstraint(Revolute(bodies[1], bodies[2],[1.0,0.0,0.0],
                    axis_offset = Dojo.RotY(π/2),
                    parent_vertex=[-radius_b,0,0],
                    child_vertex=[0,0,height_1/2]),
                    name = :joint1);  
            [JointConstraint(Revolute(bodies[i], bodies[i+1],[1.0,0.0,0.0],
                    parent_vertex=[0,0,-height_1/2],
                    child_vertex=[0,0,height_1/2]),
                    name = Symbol("joint",i)) for i = 2:num_links]]
    end
    
    
    # joints = [
    #     JointConstraint(Floating(origin, bodies[1]),
    #             name=:floating_joint),
    #     JointConstraint(Revolute(bodies[1], bodies[2],[0.0,1.0,0.0],
    #             axis_offset = Dojo.RotX(π/2),
    #             parent_vertex=[-radius_b,0,0],
    #             child_vertex=[0,0,height_1/2]),
    #             name = Symbol("joint",1)),
    #     JointConstraint(Revolute(bodies[2], bodies[3],[0.0,1.0,0.0],
    #             parent_vertex=[0,0,-height_1/2],
    #             child_vertex=[0,0,height_1/2]),
    #             name = :joint2),
    #     JointConstraint(Revolute(bodies[3], bodies[4],[0.0,1.0,0.0],
    #             parent_vertex=[0,0,-height_1/2],
    #             child_vertex=[0,0,height_2/2]),
    #             name = :joint3),
    #     JointConstraint(Revolute(bodies[4], bodies[5],[0.0,1.0,0.0],
    #             parent_vertex=[0,0,-height_2/2],
    #             child_vertex=[0,0,height_2/2]),
    #             name = :joint4)]

    mech = Mechanism(origin, bodies, joints, 
            gravity=gravity, 
            timestep=timestep, 
            spring=spring, 
            damper=damper)

    if limits
        for i=2:num_links+1
            joints[i] = Dojo.add_limits(mech, joints[i], 
                rot_limits=[SVector{1}(joint_limits[1][i-1]), SVector{1}(joint_limits[2][i-1])])
        end
        # joints[3] = Dojo.add_limits(mech, joints[3], 
        #     rot_limits=[SVector{1}(joint_limits[1][2]), SVector{1}(joint_limits[2][2])])
        # joints[4] = Dojo.add_limits(mech, joints[4], 
        #     rot_limits=[SVector{1}(joint_limits[1][3]), SVector{1}(joint_limits[2][3])])
        # joints[5] = Dojo.add_limits(mech, joints[5], 
        #     rot_limits=[SVector{1}(joint_limits[1][4]), SVector{1}(joint_limits[2][4])])

        mech = Mechanism(origin, bodies, joints, 
            gravity=gravity, 
            timestep=timestep, 
            spring=spring, 
            damper=damper)
    end
    # define the two spherical contacts, each contact is attached to a body.
    contact_point = [0.0, 0.0, 0.0] # both contact spheres are located at the center of the bodies they are attached to.
    normal = [0.0, 0.0, 1.0] # the normal of the contact is always pointing in the upright direction because the floor is flat.
    # we specify the type of contact (impact, linear, nonlinear), the coefficient of friction, the radius of the contact sphere, etc.
    if dim3
        mesh = 15
    else
        mesh = 6
    end
    
    if contact_body
        normal_mesh = [[0.0; 0.0; 1.0] for i = 1:mesh]
        friction_coefficient_mesh = friction_coefficient * ones(mesh)
        mesh_circle = [[radius_b*cos(2*π/mesh*j),radius_b/2*sin(2*π/mesh*j),0.0] for j=1:mesh]

        contacts1 = [
            contact_constraint(bodies[1], normal,
                friction_coefficient=friction_coefficient,
                contact_origin=contact_point, 
                contact_radius=height_b/2,
                contact_type=contact_type);
            # [contact_constraint(bodies[i], normal,
            #     friction_coefficient=friction_coefficient,
            #     contact_origin=[0.0,0.0,height_1/2],
            #     contact_radius=radius,
            #     contact_type=contact_type) for i=2:num_links+1];
            [contact_constraint(body, normal,
                friction_coefficient=friction_coefficient,
                contact_origin=[0.0,0.0,-height_1/2],
                contact_radius=radius,
                contact_type=contact_type) for body in bodies[3:end]];
            # contact_constraint(bodies[2], normal,
            #     friction_coefficient=friction_coefficient,
            #     contact_origin=[0.0,0.0,height_1/2],
            #     contact_radius=radius,
            #     contact_type=contact_type) 
            ]



        contacts2 = contact_constraint(fill(bodies[1],mesh), normal_mesh,
            friction_coefficient=friction_coefficient_mesh,
            contact_origins=mesh_circle,
            contact_radius=height_b/2*ones(mesh),
            contact_type=contact_type)

        mech = Mechanism(origin, bodies, joints, [contacts1;contacts2],
            gravity=gravity, 
            timestep=timestep, 
            spring=spring, 
            damper=damper)
        
    else
        mech = Mechanism(origin, bodies, joints,
            gravity=gravity, 
            timestep=timestep, 
            spring=spring, 
            damper=damper)
    end
    

    return mech
end

function initialize_pleuro!(mechanism::Mechanism; 
        body_position=[0.0; 0.0; 0.011], 
        body_orientation=[0.0; 0.0; 0.0 * π],
        stem_orientation=0.0)
      
    set_minimal_coordinates!(mechanism, get_joint(mechanism, :floating_joint), [body_position; body_orientation])
    set_minimal_coordinates!(mechanism, get_joint(mechanism, Symbol("joint1")), [0.1, stem_orientation])
    i=2
    for body in mechanism.bodies[3:end]
        if i < 4
            set_minimal_coordinates!(mechanism, get_joint(mechanism, Symbol("joint",i)), [0.0, stem_orientation])
        else
            set_minimal_coordinates!(mechanism, get_joint(mechanism, Symbol("joint",i)), [0.0, 0.0])
        end
        i+=1
    end

    zero_velocity!(mechanism)
end

