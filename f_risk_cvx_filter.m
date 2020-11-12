function [filter_risk_state] = f_risk_localization_filter(filter_state, filter_covariance, particles, weights_current, epsilon, max_iterations)
    % E[||X||^2 * X | Y]
    norm_squared_particle   = sum((vecnorm(particles).^2) .* particles .* weights_current, 2);
    
    % E[||X||^2 | Y] * \hat(X)
    norm_squared_filter     = sum((vecnorm(particles).^2) .* weights_current) * filter_state;
    
    % E[V_y (||X||^2)]
    var_squared_norm_aux    = sum((vecnorm(particles).^2) .* weights_current, 2);
    var_squared_norm        = sum((((vecnorm(particles).^2) - var_squared_norm_aux).^2) .* weights_current, 2);
    
    m_id    = eye(length(filter_covariance));
    
    % Note that CVX is unable to solve the problem due to the inverse
    % operator in the filter formula.
    cvx_begin quiet
        variable mu(1)
        maximize(mu * var_squared_norm - 2 * (filter_state + mu * ((norm_squared_particle - norm_squared_filter)' * ((m_id + 2 * mu * filter_covariance) \ (filter_state + mu * (norm_squared_particle - norm_squared_filter))))  - mu * epsilon))
        subject to
            -mu <= 0
    cvx_end
    
    filter_risk_state = (m_id + 2 * mu * filter_covariance)\(filter_state + mu * (norm_squared_particle - norm_squared_filter * filter_state) * filter_state);
end
