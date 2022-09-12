function [v_estimate, m_estimate_covariance, m_particles, v_weights] = f_particle_filter(b_opt_option, system, v_estimate_past, v_weight_past, num_particles, time_delta, effective_ratio)
    % Computes the particle filter's estimates

    % Memory allocation
    num_states  = length(system.v_state); 
    num_obs     = length(system.v_obs);   
    
    m_particles       = zeros(num_states, num_particles);
    m_dot_particles   = zeros(num_states, num_particles);
    
    v_pdf_obs_states  = zeros(1, num_particles);
    
    % Get filter parameters
    [m_noise_particles, m_noise_obs, v_mean_particles, m_c, v_obs, cov_obs] = get_parameters(b_opt_option, system, time_delta, ...    
                                                                                            v_estimate_past, num_states, num_obs, num_particles);
    % Compute particle's dinamycs
    for i = 1:num_particles
        m_dot_particles(:, i) = v_mean_particles + m_noise_particles(:, i);
        
        m_particles(:, i)     = m_dot_particles(:, i) * time_delta + v_estimate_past;
        
        v_obs_particle        = m_c * m_particles(:, i) + m_noise_obs(:, i);
        
        v_pdf_obs_states(i)   = f_pdf(v_obs, v_obs_particle, cov_obs);
    end
    
    % Compute weights
    v_pre_weights = bsxfun(@times, v_weight_past, v_pdf_obs_states);  
    v_weights     = (1 / sum(v_pre_weights)) * v_pre_weights;
    
    % Compute effective number of particles   
    effective_particles = 1 / sum(v_weights .^2);
    total_effective     = effective_ratio * num_particles;
    
    % Resample
    if effective_particles < total_effective
        [m_resample_particles, v_resample_weight] = f_resampling(m_particles, v_weights, num_particles);
        
        m_particles   = m_resample_particles;
        v_weights     = v_resample_weight;
        
    end
    
    % Compute state estimate
    v_pre_estimate  = bsxfun(@times, m_particles, v_weights);  
    v_estimate      = sum(v_pre_estimate, 2); 
    
    % Compute covariance of error estimate   
    m_estimate_covariance = (v_weights .* (m_particles - v_estimate)) * (m_particles - v_estimate)';
    
end

function [m_noise_particles, m_noise_observation, v_mean_particles, m_c, v_obs, m_cov_observation] = get_parameters(b_optimal_option, system, time_delta, ...
                                                                                                filter_state_past, num_states, num_obs, num_particles)
    % Get required parameters for the particle filter
    
    % Ger parameters from the system
    m_a = system.m_a;
    m_b = system.m_b;
    m_c = system.m_c;
    
    m_cov_system  = system.cov_system;
    m_cov_obs     = system.cov_obs;
    
    v_control   = system.v_control;
    v_conv      = system.v_conv;

    v_obs       = system.v_obs;
    
    % Computer noise, mean and covariance for particles using
    if (b_optimal_option == 0)
        
        % Sub-optimal sampling pdf
        v_mean_particles  = zeros(num_states, 1);
        m_cov_particles   = system.cov_system;
        m_noise_particles = mvnrnd(v_mean_particles, m_cov_particles, num_particles)';   
    
        v_mean_observation    = zeros(num_obs, 1);
        m_cov_observation     = system.cov_obs;
        m_noise_observation   = mvnrnd(v_mean_observation, m_cov_observation, num_particles)';  
        
        v_mean_particles      = m_a * filter_state_past + m_b * v_control + v_conv;
    
    else
        
        % Optimal sampling pdf
        m_cov_observation = m_c * m_cov_system * m_c' + m_cov_obs;
        m_cov_particles   = m_cov_system - m_cov_system * (m_c' / m_cov_observation) * m_c * m_cov_system;

        v_dot_next_state  = m_a * filter_state_past + m_b * v_control + v_conv;
        v_next_state      = v_dot_next_state * time_delta + filter_state_past;

        v_mean_observation = m_c * v_next_state; 
        v_mean_particles   = v_dot_next_state + m_cov_particles * ((m_c') / m_cov_obs) * (v_obs - v_mean_observation);

        m_noise_particles     = mvnrnd(zeros(1, length(v_mean_particles)), m_cov_particles, num_particles)';
        m_noise_observation   = mvnrnd(zeros(1, length(v_mean_observation)), m_cov_observation, num_particles)';
    end
end

function [m_resample_particles, v_resample_weight] = f_resampling(m_particles, v_weight, num_particles)
    % Resampling function
    
    % Compute cdf
    v_cumsum_weigth = cumsum(v_weight);
    
    % Initialize indices
    v_index = zeros(1, num_particles);
    j = 1; i = 1;
    
    % Roulette algorithm
    while j <= num_particles
        roulette = rand();
        
        if roulette <= v_cumsum_weigth(i)
            v_index(j) = i;
            j = j + 1;
        end
        
        i = min(num_particles, round(num_particles * rand() + 1));
    end
    
    % Resampled particles with weigths
    m_resample_particles   = m_particles(:, v_index);
    v_resample_weight      = (1 / num_particles) * ones(1, num_particles); 
end

function pdf_obs_states = f_pdf(v_observation, v_obs_particle, m_cov_obs)
    % Compute pdf using Gaussian assumption
    
    v_error = v_observation - v_obs_particle; 
    
    pdf_constant    = 1 / sqrt( det(2 * pi * m_cov_obs) );
    pdf_obs_states  = pdf_constant * exp(- 0.5 * (v_error'/m_cov_obs) * v_error);
end
