function [filter_risk_state] = f_risk_filter(filter_state, filter_covariance, particles, weights_current, epsilon, max_iterations)
    % E[||X||^2 * X | Y]
    norm_squared_particle   = sum((vecnorm(particles).^2) .* particles .* weights_current, 2);
    
    % E[||X||^2 | Y] * \hat(X)
    norm_squared_filter     = sum((vecnorm(particles).^2) .* weights_current) * filter_state;
    
    % E[V_y (||X||^2)]
    var_squared_norm_aux    = sum((vecnorm(particles).^2) .* weights_current, 2);
    var_squared_norm        = sum((((vecnorm(particles).^2) - var_squared_norm_aux).^2) .* weights_current, 2);
    
    m_id    = eye(length(filter_covariance));
    
    mu_star = f_get_mu(filter_state, filter_covariance, norm_squared_particle, norm_squared_filter, var_squared_norm, epsilon, max_iterations, m_id);
    
    filter_risk_state = (m_id + 2 * mu_star * filter_covariance) \ (filter_state + mu_star * (norm_squared_particle - norm_squared_filter));
    
end

function [mu_star] = f_get_mu(filter_state, filter_covariance, norm_squared_particle, norm_squared_filter, var_squared_norm, epsilon, max_iterations, m_id)

    mu = 1;
%     step_size = 1;
    
    objective = zeros(1,max_iterations);
    
    for i = 1:max_iterations
        grad_laplacian = g_grad_laplacian(mu, filter_state, filter_covariance, norm_squared_particle, norm_squared_filter, var_squared_norm, epsilon, m_id);
        
        step_size = backtrack(mu, filter_state, filter_covariance, norm_squared_particle, norm_squared_filter, var_squared_norm, epsilon, m_id);
        
        mu = mu + step_size * grad_laplacian;  
        
        objective(i) = f_objective(mu, filter_state, filter_covariance, norm_squared_particle, norm_squared_filter, var_squared_norm, epsilon, m_id);
    end
    
    mu_star = mu;
end

function grad_laplacian = g_grad_laplacian(mu, filter_state, filter_covariance, norm_squared_particle, norm_squared_filter, var_squared_norm, epsilon, m_id)                             
   
    grad_laplacian = var_squared_norm - 4 * ((filter_state + mu * (norm_squared_particle - norm_squared_filter))' / (m_id + 2 * mu * filter_covariance)) * ...
        ((norm_squared_particle - norm_squared_filter) - (filter_covariance / (m_id + 2 * mu * filter_covariance)) * (filter_state + mu * (norm_squared_particle - norm_squared_filter))) + 2 * epsilon;
    
end

function objective = f_objective(mu, filter_state, filter_covariance, norm_squared_particle, norm_squared_filter, var_squared_norm, epsilon, m_id)
    objective = mu * var_squared_norm - 2 * (((filter_state + mu * (norm_squared_particle - norm_squared_filter))' / (m_id + 2 * mu * filter_covariance)) * ((filter_state + mu * (norm_squared_particle - norm_squared_filter)) - epsilon * mu));
end

function step_size = backtrack(mu, filter_state, filter_covariance, norm_squared_particle, norm_squared_filter, var_squared_norm, epsilon, m_id)
    alpha   = 0.3;
    beta    = 0.4;
    
    step_size = 1;
    
    grad_laplacian = g_grad_laplacian(mu, filter_state, filter_covariance, norm_squared_particle, norm_squared_filter, var_squared_norm, epsilon, m_id);
    objective = f_objective(mu, filter_state, filter_covariance, norm_squared_particle, norm_squared_filter, var_squared_norm, epsilon, m_id);
    
    while 1
        mu_update = mu + step_size * grad_laplacian;
        objective_update = f_objective(mu_update, filter_state, filter_covariance, norm_squared_particle, norm_squared_filter, var_squared_norm, epsilon, m_id);
        condition_eval = objective + alpha * step_size * (grad_laplacian') * grad_laplacian;
        if (objective_update > condition_eval)
            step_size = beta * step_size;
        else
            break;
        end
    end
end