function [filter_state, filter_covariance, particles, weights_current] = f_opt_filter(system, filter_state_past, weight_past, num_particles, time_delta)
    % NOT REQUIRED!!!!
    
    % DELETE!!!
    
    
    % Memory allocation
    num_states = length(system.v_state);                                          
    particles = zeros(num_states, num_particles);
    
    dot_particles = zeros(num_states, num_particles);
    
    pdf_obs_states = zeros(1, num_particles);
    
    % Get system parameters
    m_a = system.m_a;
    m_b = system.m_b;
    m_c = system.m_c;
    
    cov_system = 1.5 * system.cov_system;
    cov_obs = 1.5 * system.cov_obs;
    
    v_control = system.v_control;
    v_conv = system.v_conv;
   
    v_obs = system.v_obs;
    
    % Optimal important sampling pdf
    cov_observation = m_c * cov_system * m_c' + cov_obs;
    cov_particles   = cov_system - cov_system * (m_c' / cov_observation) * m_c * cov_system;
    
    dot_next_state = m_a * filter_state_past + m_b * v_control + v_conv;
    next_state = dot_next_state * time_delta + filter_state_past;
    
    mean_observation = m_c * next_state;
    mean_particles   = dot_next_state + cov_particles * ((m_c') / cov_obs) * (v_obs - mean_observation);
    
    noise_particles = mvnrnd(zeros(1, length(mean_particles)), cov_particles, num_particles)';
    noise_observation = mvnrnd(zeros(1, length(mean_observation)), cov_observation, num_particles)';
    
    for i = 1:num_particles
        
        dot_particles(:, i) = mean_particles + noise_particles(:, i);
        
        particles(:, i) = dot_particles(:, i) * time_delta + filter_state_past;
        
        obs_particle = mean_observation + noise_observation(:, i);
        
        pdf_obs_states(i) = f_pdf(v_obs, obs_particle, cov_observation);
    end
    
    pre_weights_current = bsxfun(@times, weight_past, pdf_obs_states);  
    weights_current = bsxfun(@rdivide, pre_weights_current, sum(pre_weights_current)); 
    
    effective_particles = 1 / sum(weights_current .^2);
    total_effective = 0.5 * num_particles;
    
    if effective_particles < total_effective
        [res_particles, res_weight] = f_resampling(particles, weights_current, num_particles);
        particles = res_particles;
        weights_current = res_weight;
    end
    
    pre_filter_state = bsxfun(@times, particles, weights_current);  
    filter_state = sum(pre_filter_state, 2); 
    
    deviation_particles = particles - filter_state; % fix this
    filter_covariance = ((deviation_particles .* weights_current) * deviation_particles'); % double check if we need 1/Number_particles
    
end

function [res_particles, res_weight] = f_resampling(particles, weight, num_particles)
    
    cumsum_weigth = cumsum(weight);
    
    index = zeros(1, num_particles);
    j = 1; i = 1;
    
    while j <= num_particles
        roulette = rand();
        
        if roulette <= cumsum_weigth(i)
            index(j) = i;
            j = j + 1;
        end
        
        i = min(num_particles, round(num_particles * rand() + 1));
    end
    
    res_particles = particles(:, index);
    res_weight = bsxfun(@rdivide, weight(index), sum(weight(index)));
end


function pdf_obs_states = f_pdf(observation, obs_particle, cov_observation)

    variance_noise = cov_observation; 
    
    error = observation - obs_particle; 
    
    pdf_constant = 1/sqrt(det(variance_noise)*(2*pi)^(1));
    pdf_obs_states = pdf_constant*exp(-0.5*(error'/variance_noise)*error);
end
