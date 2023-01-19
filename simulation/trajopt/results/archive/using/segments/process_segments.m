clear
close all
files = dir('x*.mat');
files = natsortfiles(files);

scale=1;
num_links = 3;
radius = 0.01*scale;
radius_b = 0.05*scale;

height_b = 2*radius;
timestep = 0.05;
T = 101;
time = timestep*T;
num_u = zeros(length(files),1);
avg_velocity = zeros(length(files),1);
COT = zeros(length(files),1);
gait = {};
times=0:0.05:100*0.05;
for i=1:length(files)
    num_u = str2double(files(i).name(end-6));

    g_string = files(i).name(21);
    h_total = 0.15;
    height = h_total/2/num_links;
    mass = pi*radius^2*height*1500;
    mass_2 = mass/2;
    mass_b = pi*radius_b^2*height_b*1500;
    mass_total = (mass*num_links + mass_2*num_links + mass_b);
    Fg = 9.81*mass_total;
    load(files(i).name)
    u_name = strcat('u', files(i).name(2:end));
    load(u_name)
    avg_velocity(i) = -x_sol{end}(1)/time*100;
    power = 0;
    for t=1:T-1
        power = power + 600000*u_sol{t}'* u_sol{t};
    end
    COT(i) = power/Fg/-x_sol{end}(1);
    xmat = [x_sol{:}]';
    umat = [u_sol{:}]';
    
    figure
    yyaxis left
    plot(times,-xmat(:,1)*100,"LineWidth",2)
    xlabel("Time (s)")
    ylabel("Distance (cm)")
    hold on
    axis([0 5.15 0 3.5])
    yyaxis right
    p = [];
    for j=1:num_u
        p(j) = plot(times(1:end-1),umat(:,j)/timestep,"LineWidth",1.25);
    end
    if num_u == 3
        legend(p, "u_1","u_2","u_3")
    else
        legend(p, "u_1","u_2","u_3","u_4")
    end
    ylabel("Torque (Nm)")
    set(gca,'FontSize',15)
end

% gait = gait';
% C = linspecer(6);
% gscatter(c,avg_velocity, gait,C,[],40)
% xlabel('Cost scaling of inputs')
% ylabel('Average Velocity (cm/s)')
% set(gca,'FontSize',15)
% set(gca,'xscale','log') 
% figure
% gscatter(c,COT, gait,C,[],40)
% xlabel('Cost scaling of inputs')
% ylabel("Cost of Transport")
% set(gca,'FontSize',15)
% set(gca,'xscale','log')
% set(gca,'yscale','log')