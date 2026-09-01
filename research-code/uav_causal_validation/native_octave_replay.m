function native_octave_replay(workspace_path, output_path)
% Fail-closed replay of the archived EPFL controller through the pinned
% acados MATLAB/Octave interface.  No recorded intermediate state or input is
% supplied to the controller.
d = load(workspace_path);
S = d.S;
p_swarm = S;
p_swarm.v_ref = S.v_swarm;
p_swarm.u_ref = double(S.u_migration(:));
p_swarm.d_ref = S.d;
p_swarm.Pos0 = reshape(double(d.pos_history(1,:)), 3, S.nb_agents);
p_swarm.Vel0 = reshape(double(d.vel_history(1,:)), 3, S.nb_agents);
map = d.map;
dt = median(diff(double(d.time_history(:))));
T = double(d.time_history(end) - d.time_history(1));
final_positions = reshape(double(d.pos_history(end,:)),3,S.nb_agents);
end_line = min(final_positions(1,:));
model = []; ocp = []; sim = [];
[model, ocp, sim, map, p_swarm, p, nb_steps_sim, x0] = ...
  cl_init_fun(dt, T, model, ocp, sim, map, p_swarm);
[x_history, u_history, status, sqp_iter, time_tot, time_lin, time_qp_sol] = ...
  cl_run_fun(model, ocp, sim, map, p_swarm, dt, p, nb_steps_sim, x0, end_line);
pos_history = x_history(1:3*S.nb_agents,:)';
vel_history = x_history(3*S.nb_agents+1:end,:)';
U_history = u_history';
time_history = (0:size(pos_history,1)-1)' * dt;
save('-mat7-binary', output_path, 'pos_history', 'vel_history', 'U_history', ...
     'time_history', 'status', 'sqp_iter', 'time_tot', 'time_lin', 'time_qp_sol');
end
