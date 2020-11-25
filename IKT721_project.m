function IKT721_project()
    % Main function
    
    rng('default'); 

    import aquaponics;
    
    %% Parameters
    
    % Time [minutes]
    time_start = 0; time_end = 720; time_delta = 1; 
    v_time_index = time_start:time_delta:time_end;

    % Particle filter parameters
    num_particles   = 1000;
    effective_ratio = 0.5;
    
    % Risk-aware filter parameter
    epsilon         = 1.65e2;
    max_iterations  = 50;
    
    % Memory allocation
    num_states  = 4;
    steps       = [num_states (length(v_time_index)-1)];
    
    % Estimated values
    m_estimate_sub  = zeros(steps);
    m_estimate_opt  = zeros(steps);
    m_estimate_risk = zeros(steps);
    
    % Risk history
    v_risk_sub      = zeros(1, length(v_time_index)-1);
    v_risk_opt      = zeros(1, length(v_time_index)-1);
    v_risk_risk     = zeros(1, length(v_time_index)-1);
    
    % Computational time
    v_time_sub      = zeros(1, length(v_time_index)-1);
    v_time_opt      = zeros(1, length(v_time_index)-1);
    v_time_risk     = zeros(1, length(v_time_index)-1);
    
    % Simulation parameters
    num_simulation  = 100; 
    
    % Simulation estimations
    m_estimate_sub_sim  = zeros([steps, num_simulation]);
    m_estimate_opt_sim  = zeros([steps, num_simulation]);
    m_estimate_risk_sim = zeros([steps, num_simulation]);
    
    % Simulation risk values
    m_risk_sub_sim      = zeros([length(v_time_index)-1, num_simulation]);
    m_risk_opt_sim      = zeros([length(v_time_index)-1, num_simulation]);
    m_risk_risk_sim     = zeros([length(v_time_index)-1, num_simulation]);
    
    % Simulation MSE values
    m_mse_sub_sim       = zeros([length(v_time_index)-1, num_simulation]);
    m_mse_opt_sim       = zeros([length(v_time_index)-1, num_simulation]);
    m_mse_risk_sim      = zeros([length(v_time_index)-1, num_simulation]);   
    
    %% Simulation
    for j = 1:num_simulation
        % Initialize biofilter class
        system = aquaponics();
        
        % Particle filter initialization
        v_weight_past_sub = repmat(1 / num_particles, [1, num_particles]);
        v_weight_past_opt = v_weight_past_sub;
    
        v_estimate_past_sub = system.v_state;
        v_estimate_past_opt = v_estimate_past_sub;
        
        
        for i = v_time_index(2:end)

            % Control inputs
            feed = 2000;
            nhn_hyd = 45; 

            % Update system
            system = system.f_update_dynamics(time_delta, i, feed, nhn_hyd);

            % Suboptimal particle filter
            tic;
            b_opt_option = 0;
            [m_estimate_sub(:, i), ~ , m_particles_sub, v_weights_sub] = f_particle_filter(b_opt_option, system, v_estimate_past_sub, v_weight_past_sub, num_particles, time_delta, effective_ratio);
            v_risk_sub(i) = f_get_predictive_variance(m_particles_sub, m_estimate_sub(:, i), v_weights_sub);
            v_time_sub(i) = toc;

            % Optimal particle filter
            tic;
            b_opt_option = 1;
            [m_estimate_opt(:, i), m_estimate_covariance_opt, m_particles_opt, v_weights_opt] = f_particle_filter(b_opt_option, system, v_estimate_past_opt, v_weight_past_opt, num_particles, time_delta, effective_ratio);
            v_risk_opt(i) = f_get_predictive_variance(m_particles_opt, m_estimate_opt(:, i), v_weights_opt);
            v_time_opt(i) = toc;

            % Risk-aware filter: ADMM method with optimal sampling
            [m_estimate_risk(:, i)]   = f_risk_filter_ADMM(m_estimate_opt(:, i), m_estimate_covariance_opt, m_particles_opt, v_weights_opt, epsilon, max_iterations);
            v_risk_risk(i) = f_get_predictive_variance(m_particles_opt, m_estimate_risk(:, i), v_weights_opt);
            v_time_risk(i) = toc;

            % Save current filtered states
            v_estimate_past_sub = m_estimate_sub(:, i);
            v_estimate_past_opt = m_estimate_opt(:, i);

            % Save current weights
            v_weight_past_sub = v_weights_sub;
            v_weight_past_opt = v_weights_opt;
        end
        
        % Save estimate values for jth simulation
        m_estimate_sub_sim(:, :, j)  = m_estimate_sub;
        m_estimate_opt_sim(:, :, j)  = m_estimate_opt;
        m_estimate_risk_sim(:, :, j) = m_estimate_risk;
        
        % Save Risk results for jth simulation
        m_risk_sub_sim(:, j)    = v_risk_sub;
        m_risk_opt_sim(:, j)    = v_risk_opt;
        m_risk_risk_sim(:, j)   = v_risk_risk;

        % Compute MSE for results
        [v_mse_sub, v_mse_opt, v_mse_risk] = f_get_mse(system, m_estimate_sub, m_estimate_opt, m_estimate_risk);
        
        % Save MSE results for jth simulation
        m_mse_sub_sim(:, j) = v_mse_sub;
        m_mse_opt_sim(:, j) = v_mse_opt;
        m_mse_risk_sim(:, j)= v_mse_risk;
        
    end
    
    %% Results
    
    m_avg_estimate_sub  =  mean(m_estimate_sub, 3);
    m_avg_estimate_opt  =  mean(m_estimate_opt, 3);
    m_avg_estimate_risk =  mean(m_estimate_risk, 3);
    
    % Compute average risk measure
    [v_avg_risk_sub, v_avg_risk_opt, v_avg_risk_risk] = f_get_average(m_risk_sub_sim, m_risk_opt_sim, m_risk_risk_sim);
    
    % Compute average MSE
    [v_avg_mse_sub, v_avg_mse_opt, v_avg_mse_risk]    = f_get_average(m_mse_sub_sim, m_mse_opt_sim, m_mse_risk_sim);
    
    % Compute average computational time
    f_get_average(v_time_sub, v_time_opt, v_time_risk);
    
    % Save variables in workspace
    save('results5.mat')
    
    % Print results
%     metric = 'Computational Time [seconds]';
%     f_print_results(metric, v_avg_sub, v_avg_opt, v_avg_risk)
    
%     % Plot risk for results
%     f_plot_risk(v_time_index, v_risk_sub, v_risk_opt, v_risk_risk);
%     
%     % Plot MSE for results
%     f_plot_mse(v_time_index, v_mse_sub, v_mse_opt, v_mse_risk);
%     
%     % Plot estimates and true states
%     f_plot_estimates(system, m_estimate_sub, m_estimate_opt, m_estimate_risk);
    
end

%% Auxiliar function

function [v_avg_sub, v_avg_opt, v_avg_risk] = f_get_average(v_sub, v_opt, v_risk)
    % Compute average metrics: computational time, mse, risk
    
    v_avg_sub    =  mean(v_sub, 2);
    v_avg_opt    =  mean(v_opt, 2);
    v_avg_risk   =  mean(v_risk, 2);
    
end

function f_print_results(metric, v_avg_sub, v_avg_opt, v_avg_risk)
    % Print results in the terminal
    
    fprintf('------------Average %s----------------\n', metric);
    fprintf('Average %s for suboptimal particle filter: %d\n', metric, v_avg_sub);
    fprintf('Average %s for optimal particle filter: %d\n', metric, v_avg_opt);
    fprintf('Average %s for risk aware mmse: %d\n', metric, v_avg_risk);
end

function f_plot_estimates(time_index, system, filter_state, opt_filter_state, filter_risk_state)
    % Plot the resulting estimates and compares them with the real value
    
    % Plot attributes
    legend_label    = {'True Value', 'Suboptimal PF', 'Optimal PF', 'Risk-aware MMSE'};
    title_label     = {'NH-N tank 1', 'NH-N tank 2', 'NO3-N tank 1', 'NO3-N tank 2'};
    x_label         = {'Time [min]'};
    y_label         = {'Concentration [g l^{-1}]'};
    
    % Plotting
    for i = 1:4
        figure; 
        plot(time_index(2:end), system.v_state_history(i, 2:end), '.-k', time_index(2:end), filter_state(i, :), 'r', ...
             time_index(2:end), opt_filter_state(i, :), 'b', time_index(2:end), filter_risk_state(i, :), 'm', 'LineWidth', 1);
        title(title_label(i)); 
        legend(legend_label);
        xlabel(x_label); ylabel(y_label); xlim([0 max(time_index)]);
        grid on; grid minor;
    end
end

function risk = f_get_predictive_variance(particles, estimate, weights)
    % E[ V_y (||X - \hat{X}||^2) ]
    m_error               = particles - estimate;
    expected_error      = sum((vecnorm(m_error).^2) .* weights, 2);
    risk = sum((((vecnorm(m_error).^2) - expected_error).^2) .* weights, 2);
end

function f_plot_risk(v_time_index, v_risk_sub, v_risk_opt, v_risk_risk)

    % Plot the results
    figure;
    plot(v_time_index(2:end), v_risk_sub, 'r', v_time_index(2:end), v_risk_opt, 'b', v_time_index(2:end), v_risk_risk, 'm', 'LineWidth', 1);
    title('Risk');
    legend({'Suboptimal PF', 'Optimal PF', 'Risk-aware MMSE'})
    xlabel('Time [min]'); xlim([0 max(v_time_index)]);
    ylabel('E[ V_y (||X - X_{est}||^2) ]');
    grid on; grid minor;

end

function [v_mse_sub, v_mse_opt, v_mse_risk] = f_get_mse(system, m_estimate_sub, m_estimate_opt, m_estimate_risk)
    % Compute and plot mse between estimate and true valie
    
    % Compute errors
    m_error_filter        = system.v_state_history(:, 2:end) - m_estimate_sub;
    m_error_opt_filter    = system.v_state_history(:, 2:end) - m_estimate_opt;
    m_error_filter_risk   = system.v_state_history(:, 2:end) - m_estimate_risk;
    
    % Compute MSE 
    v_mse_sub    = vecnorm(m_error_filter) .^ 2;
    v_mse_opt    = vecnorm(m_error_opt_filter) .^ 2;
    v_mse_risk   = vecnorm(m_error_filter_risk) .^ 2;
end

function f_plot_mse(v_time_index, v_mse_sub, v_mse_opt, v_mse_risk)
    % Plot MSE
    figure;
    plot(v_time_index(2:end), v_mse_sub , 'r', v_time_index(2:end), v_mse_opt, 'b',...
         v_time_index(2:end), v_mse_risk, 'm', 'LineWidth', 1);
    title('MSE');
    legend({'Suboptimal PF', 'Optimal PF', 'Risk-aware MMSE'})
    xlabel('Time [min]'); xlim([0 max(v_time_index)]);
    ylabel('MSE'); 
    grid on; grid minor;
end