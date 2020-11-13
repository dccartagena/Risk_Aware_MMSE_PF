function v_estimate = f_risk_filter_ADMM(v_estimate, m_estimate_covariance, m_particles, v_weights, epsilon, max_iterations)
    % Compute the risk-aware estimate
    
    % E[ ||X||^2 * X | Y ]
    v_norm_squared_particle   = sum((vecnorm(m_particles).^2) .* m_particles .* v_weights, 2);
    
    % E[||X||^2 | Y] * E[ X | Y ]
    norm_squared_filter     = sum((vecnorm(m_particles).^2) .* v_weights) * v_estimate;
    
    % E[ V_y (||X||^2) ]
    var_squared_norm_aux    = sum((vecnorm(m_particles).^2) .* v_weights, 2);
    var_squared_norm        = sum((((vecnorm(m_particles).^2) - var_squared_norm_aux).^2) .* v_weights, 2);
    
    % Compute risk-aware estimate via ADMM
    v_estimate = f_admm(v_estimate, m_estimate_covariance, v_norm_squared_particle, ...
                        norm_squared_filter, var_squared_norm, epsilon, max_iterations);
    
end

function v_estimate_admm = f_admm(v_estimate, m_estimate_covariance, v_norm_squared_particle, ...
                                norm_squared_filter, var_squared_norm, epsilon, max_iterations)
    % Consensus-ADMM optimizer algorithm

    % Inicialization
    rho             = 0.5;
    v_estimate_admm = v_estimate;
    v_consensus     = zeros(size(v_estimate));
    v_dual_variable = 1e-3;
    
    % ADMM algorithm
    for i = 1:max_iterations
        
        v_estimate_admm  = ( v_estimate_admm + rho * (v_consensus + v_dual_variable) ) / (1 + rho) ;
        v_consensus     = f_bisection(v_estimate_admm, m_estimate_covariance, v_dual_variable, ...
                                        v_norm_squared_particle, norm_squared_filter, var_squared_norm, epsilon, max_iterations);
        v_dual_variable = v_dual_variable + v_consensus - v_estimate_admm;
        
    end
end

function v_consensus = f_bisection(v_estimate_admm, m_estimate_covariance, v_dual_variable, ...
                                    v_norm_squared_particle, norm_squared_filter, var_squared_norm, epsilon, max_iterations)
    % Bisection algorithm
    
    % Parameter
    tolerance           = 1e-3;
    convergence_value   = 1;
    iteration           = 1;
    
    % Decompose covariance matrix
    [m_decomp_cov, m_eig_cov]   = eig(m_estimate_covariance);
    v_eig_cov                   = diag(m_eig_cov);
    
    % Inicialization
    min_eig_cov     = min(v_eig_cov);
    max_eig_cov     = max(v_eig_cov);
    v_dual_newton   = - (min_eig_cov + max_eig_cov) / (2 * min_eig_cov * max_eig_cov);
    
    % Compute auxiliar variables to solve equation via Newton method
    v_tilde_chi = m_decomp_cov' * (v_estimate_admm - v_dual_variable);
    v_tilde_b   = m_decomp_cov' * 0.5 * (v_norm_squared_particle - norm_squared_filter);
    v_c         = 0.25 * (epsilon - var_squared_norm);
    
    % Bisection method for solving gradient of lagrangian = 0   
    max_mu =  100;
    min_mu = -100;
    
    if (min_eig_cov < 0)
        max_mu = - 1 / min_eig_cov;
    end
    
    if (max_eig_cov > 0)
        min_mu = - 1 / max_eig_cov;
    end
    
    while ((convergence_value > tolerance) && (iteration < max_iterations))
        v_dual_newton   = (max_mu + min_mu) / 2;
        phi_value       = f_phi_equation(v_dual_newton, v_eig_cov, v_tilde_chi, v_tilde_b, v_c);
        
        if phi_value > 0
            min_mu = v_dual_newton;
        else
            max_mu = v_dual_newton;
        end
        
        convergence_value   = max_mu - min_mu;
        iteration           = iteration + 1;
    end
    
    % Compute consensus variable for ADMM
    m_eye       = eye(length(v_eig_cov));
    v_consensus = m_decomp_cov * ((m_eye + v_dual_newton * m_eig_cov) \ (v_tilde_chi + v_dual_newton * v_tilde_b) );
end

function phi_value = f_phi_equation(v_dual_newton, v_eig_cov, v_tilde_chi, v_tilde_b, v_c)
    % Compute target function for bisection method
    
    phi_value = 0; 
    
    for k = 1:length(v_eig_cov)
        
        phi_value = phi_value + v_eig_cov(k) * ((v_tilde_chi(k) + v_dual_newton * v_tilde_b(k)) / (1 + v_dual_newton * v_eig_cov(k))) .^ 2;
        phi_value = phi_value - 2 * (v_tilde_b(k) * (v_tilde_chi(k) + v_dual_newton * v_tilde_b(k)) / (1 + v_dual_newton * v_eig_cov(k)));
    
    end
    
    phi_value = phi_value - v_c;   
end




































