function [m_crlb] = f_lower_bound(system, m_rclb_past)
    % Computes the Cramer-Rao Lower Bound (CRLB)
    
    % System parameters
    m_a = system.m_a;
    m_c = system.m_c;
    
    cov_system = system.cov_system;
    cov_obs = system.cov_obs;
    
    conv_rate = system.conv_rate;
    v_state_2 = system.v_state(2);
    
    % Gradient for non-linear dynamics
    grad_v_conv =  [0 0 0 0; 
                    0 (1 / system.volume_tank2)*(-2 * (conv_rate / (conv_rate + 2 * v_state_2)) .^ 2) 0 0; 
                    0 0                                                                               0 0; 
                    0 (1 / system.volume_tank2)*( 2 * (conv_rate / (conv_rate + 2 * v_state_2)) .^ 2) 0 0];
    
    % Gradient full dymanics
    grad_dynamics = m_a + grad_v_conv; % DOUBLE CHECK THIS

    % CRLB equations
    crlb_11 = (grad_dynamics / cov_system) * grad_dynamics';
    crlb_12 = grad_dynamics' / cov_system;
    crlb_21 = crlb_12';
    crlb_22 = eye(size(cov_system))/cov_system + (m_c' * (cov_obs \ m_c));
    
    m_crlb = crlb_22 - (crlb_21 / (m_rclb_past + crlb_11)) * crlb_12;  
    
end