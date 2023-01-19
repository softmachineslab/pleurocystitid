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
c = zeros(length(files),1);
avg_velocity = zeros(length(files),1);
COT = zeros(length(files),1);
gait = {};

for i=1:length(files)
    c_string = files(i).name(end-9);
    switch c_string
        case '1'
            c(i) = 1/100;
        case '2'
            c(i) = 1/10;
        case '3'
            c(i) = 1;
        case '4'
            c(i) = 10;
        case '5'
            c(i) = 100;
    end
    g_string = files(i).name(21);
    switch g_string
        case 'c'
            gait{i} = 'Sculling';
        case 'i'
            gait{i} = 'Sinusoidal';
    end
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
        power = power + 1000*u_sol{t}'* u_sol{t};
    end
    COT(i) = power/Fg/-x_sol{end}(1);
    
    
    
end

gait = gait';
C = linspecer(6);
gscatter(c,avg_velocity, gait,C,[],40)
xlabel('Cost scaling of inputs')
ylabel('Average Velocity (cm/s)')
set(gca,'FontSize',15)
set(gca,'xscale','log') 
figure
gscatter(c,COT, gait,C,[],40)
xlabel('Cost scaling of inputs')
ylabel("Cost of Transport")
set(gca,'FontSize',15)
set(gca,'xscale','log')
set(gca,'yscale','log')