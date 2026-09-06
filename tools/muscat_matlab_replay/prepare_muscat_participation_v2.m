% Generate a preregistered MuSCAT source-class participation matrix.
% Scientific constants are fixed here before the cloud run.  The matrix is
% deliberately small enough for an unmodified MATLAB source integration.
root = getenv('GITHUB_WORKSPACE');
source_file = fullfile(root, 'external', 'muscat', 'Mission', 'Mission_NISAR_2SC.m');
main_file = fullfile(root, 'external', 'muscat', 'Main', 'main_v3.m');
artifact_dir = fullfile(root, 'artifacts');
if ~isfolder(artifact_dir), mkdir(artifact_dir); end

source = fileread(source_file);
main = fileread(main_file);
assert(count(source, 'init_data.num_hardware_exists.num_reaction_wheel = 0;') == 1);
source = replace(source, 'init_data.num_hardware_exists.num_reaction_wheel = 0;', ...
    'init_data.num_hardware_exists.num_reaction_wheel = 4;');
source = replace(source, '% init_data.time_step_attitude = 5;', 'init_data.time_step_attitude = 1;');
source = replace(source, 'init_data.flag_realtime_plotting = 1;', 'init_data.flag_realtime_plotting = 0;');
source = replace(source, 'init_data.flag_save_plots = 1;', 'init_data.flag_save_plots = 0;');
source = replace(source, 'init_data.t_final = 8*86400;', 'init_data.t_final = 180;');
source = replace(source, 'init_data.t_final = 4*86400;', 'init_data.t_final = 180;');
source = replace(source, 'init_data.t_final = 1*86400;', 'init_data.t_final = 180;');

old_call = 'func_main_software_SC_control_attitude(mission.true_SC{i_SC}.software_SC_control_attitude, mission, i_SC);';
assert(count(main, old_call) == 1);
main = replace(main, old_call, 'func_muscat_heterogeneous_attitude_control(mission, i_SC);');
custom_main = fullfile(root, 'external', 'muscat', 'Main', 'main_muscat_heterogeneous.m');
writelines(string(main), custom_main);

start_marker = '%% Save All Data';
first_save = strfind(source, start_marker);
assert(numel(first_save) == 2 && count(source, 'run main_v3.m') == 1);
prefix = source(1:first_save(1)-1);

rho_grid = [0.10 0.20 0.40 0.80 1.60 2.40];
inertia_multiplier = [0.92 1.06; 0.95 1.08; 0.93 1.04];
wheel_limit = [0.18 0.22; 0.19 0.215; 0.185 0.225];
bias_z = [0.010 0.014; 0.012 0.009; 0.008 0.016];
observer_z0 = [0.00035 -0.00030; 0.00042 -0.00024; 0.00029 -0.00038];
policies = {'permanent','two_layer'};

driver = fullfile(root, 'external', 'muscat', 'Mission', 'run_muscat_participation_v2.m');
fid_driver = fopen(driver, 'w'); assert(fid_driver > 0);
fprintf(fid_driver, '%% Generated frozen participation driver\n');

for s = 1:3
    for r = 1:numel(rho_grid)
        for p = 1:numel(policies)
            policy = policies{p}; rho = rho_grid(r);
            tag = sprintf('s%02d_r%02d_%s', s, r, policy);
            setup = sprintf([ ...
                '\n%% Frozen MuSCAT participation V2 condition\n', ...
                'global MUSCAT_HETERO\n', ...
                'MUSCAT_HETERO = struct(); MUSCAT_HETERO.k=0; MUSCAT_HETERO.policy=''%s'';\n', ...
                'MUSCAT_HETERO.rho=%.17g; MUSCAT_HETERO.observer_gain=0.08; MUSCAT_HETERO.coupling_gain=0.06;\n', ...
                'MUSCAT_HETERO.target_rate=2e-4; MUSCAT_HETERO.target_frequency=0.025;\n', ...
                'MUSCAT_HETERO.gate_threshold=2.5e-4; MUSCAT_HETERO.pin=[1;0];\n', ...
                'MUSCAT_HETERO.A_info=[0 0;1 0]; MUSCAT_HETERO.A_physical=[0 1;1 0];\n', ...
                'MUSCAT_HETERO.kd=[8500;10500]; MUSCAT_HETERO.bias_torque=[0 0 0;0 0 %.17g];\n', ...
                'MUSCAT_HETERO.z=[0 0 %.17g;0 0 %.17g];\n', ...
                'MUSCAT_HETERO.target=zeros(1000,3); MUSCAT_HETERO.omega=zeros(1000,2,3);\n', ...
                'MUSCAT_HETERO.z_store=zeros(1000,2,3); MUSCAT_HETERO.residual=zeros(1000,2);\n', ...
                'MUSCAT_HETERO.edge_count=zeros(1000,1); MUSCAT_HETERO.applied_proxy=zeros(1000,2,3);\n', ...
                'mission.true_SC{1}.true_SC_body.total_MI = %.17g*mission.true_SC{1}.true_SC_body.total_MI;\n', ...
                'mission.true_SC{2}.true_SC_body.total_MI = %.17g*mission.true_SC{2}.true_SC_body.total_MI;\n', ...
                'lims=[%.17g %.17g]; for hs=1:mission.num_SC, for hw=1:4, mission.true_SC{hs}.true_SC_reaction_wheel{hw}.maximum_torque=lims(hs); mission.true_SC{hs}.true_SC_reaction_wheel{hw}.maximum_acceleration=lims(hs)/mission.true_SC{hs}.true_SC_reaction_wheel{hw}.moment_of_inertia; end, end\n', ...
                'run(''main_muscat_heterogeneous.m'');\n', ...
                'k=MUSCAT_HETERO.k; rows=zeros(k*2,16); q=0;\n', ...
                'for kk=1:k, for hs=1:2, q=q+1; rows(q,:)=[%d,%.17g,kk,hs,MUSCAT_HETERO.target(kk,:),reshape(MUSCAT_HETERO.omega(kk,hs,:),1,3),reshape(MUSCAT_HETERO.z_store(kk,hs,:),1,3),MUSCAT_HETERO.residual(kk,hs),MUSCAT_HETERO.edge_count(kk),norm(reshape(MUSCAT_HETERO.applied_proxy(kk,hs,:),1,3))]; end, end\n', ...
                'assert(all(isfinite(rows),''all'') && any(rows(:,16)>0));\n', ...
                'writematrix(rows,fullfile(getenv(''GITHUB_WORKSPACE''),''artifacts'',''muscat_participation_%s.csv''));\n'], ...
                policy, rho, bias_z(s,2), observer_z0(s,1), observer_z0(s,2), ...
                inertia_multiplier(s,1), inertia_multiplier(s,2), wheel_limit(s,1), ...
                wheel_limit(s,2), s, rho, tag);
            out = fullfile(root, 'external', 'muscat', 'Mission', ['Mission_MUSCAT_PART_', tag, '.m']);
            fid=fopen(out,'w'); assert(fid>0); fwrite(fid,[prefix,setup]); fclose(fid);
            fprintf(fid_driver, 'run(''%s'');\n', out);
        end
    end
end
fclose(fid_driver);

