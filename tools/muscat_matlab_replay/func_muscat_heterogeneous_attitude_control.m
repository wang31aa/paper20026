function func_muscat_heterogeneous_attitude_control(mission, i_SC)
% Source-class causal controller for the frozen designed-heterogeneity test.
% It commands the existing MuSCAT reaction-wheel objects; it never edits a
% previously generated trajectory.
global MUSCAT_HETERO

n = mission.num_SC;
if i_SC == 1
    MUSCAT_HETERO.k = MUSCAT_HETERO.k + 1;
    k = MUSCAT_HETERO.k;
    dt = mission.true_time.time_step_attitude;
    omega = zeros(n,3);
    for j = 1:n
        omega(j,:) = mission.true_SC{j}.true_SC_adc.angular_velocity;
    end
    target = [0, 0, MUSCAT_HETERO.target_rate * sin(MUSCAT_HETERO.target_frequency * mission.true_time.time_attitude)];
    L = diag(sum(MUSCAT_HETERO.A_info,2)) - MUSCAT_HETERO.A_info;
    pin = MUSCAT_HETERO.pin(:);
    MUSCAT_HETERO.z = MUSCAT_HETERO.z + dt * MUSCAT_HETERO.observer_gain * MUSCAT_HETERO.rho * ...
        (-L * MUSCAT_HETERO.z + pin * target - pin .* MUSCAT_HETERO.z);
    residual = vecnorm(omega - MUSCAT_HETERO.z, 2, 2);
    edge_weight = MUSCAT_HETERO.A_physical;
    if strcmp(MUSCAT_HETERO.policy, 'two_layer')
        trusted = double(residual <= MUSCAT_HETERO.gate_threshold);
        trusted(pin > 0) = 1;
        edge_weight = edge_weight .* trusted';
    end
    coupling = zeros(n,3);
    for j = 1:n
        coupling(j,:) = sum(edge_weight(j,:)'.*(omega-omega(j,:)),1);
    end
    requested = -MUSCAT_HETERO.kd .* (omega-MUSCAT_HETERO.z) + ...
        MUSCAT_HETERO.rho * MUSCAT_HETERO.coupling_gain .* coupling + MUSCAT_HETERO.bias_torque;
    MUSCAT_HETERO.requested = requested;
    MUSCAT_HETERO.target(k,:) = target;
    MUSCAT_HETERO.omega(k,:,:) = omega;
    MUSCAT_HETERO.z_store(k,:,:) = MUSCAT_HETERO.z;
    MUSCAT_HETERO.residual(k,:) = residual;
    MUSCAT_HETERO.edge_count(k,1) = nnz(edge_weight);
end

tau = MUSCAT_HETERO.requested(i_SC,:)';
B = zeros(3, mission.true_SC{i_SC}.true_SC_body.num_hardware_exists.num_reaction_wheel);
for j = 1:size(B,2)
    wheel = mission.true_SC{i_SC}.true_SC_reaction_wheel{j};
    B(:,j) = wheel.orientation' * wheel.moment_of_inertia;
end
alpha = pinv(B) * tau;
for j = 1:numel(alpha)
    wheel = mission.true_SC{i_SC}.true_SC_reaction_wheel{j};
    wheel.flag_executive = true;
    wheel.commanded_angular_acceleration = alpha(j);
end
limits = arrayfun(@(j) mission.true_SC{i_SC}.true_SC_reaction_wheel{j}.maximum_acceleration, 1:numel(alpha))';
tau_proxy = B * max(min(alpha, limits), -limits);
MUSCAT_HETERO.applied_proxy(MUSCAT_HETERO.k,i_SC,:) = reshape(tau_proxy,1,1,3);
end
