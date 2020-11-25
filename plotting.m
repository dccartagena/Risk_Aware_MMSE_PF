clear all; clc;
load('results5.mat')

% Plot attributes
legend_label    = {'True Value', 'Suboptimal PF', 'Optimal PF', 'Risk-aware MMSE'};
title_label     = {'NH-N tank 1', 'NH-N tank 2', 'NO3-N tank 1', 'NO3-N tank 2'};
x_label         = {'Time [min]'};
y_label         = {'Concentration [g l^{-1}]'};

% Plot estimates 
for i = 1:4
    figure; 
    plot(v_time_index(2:end), system.v_state_history(i, 2:end), '.-k', v_time_index(2:end), m_avg_estimate_sub(i, :), 'r', ...
         v_time_index(2:end), m_avg_estimate_opt(i, :), 'b', v_time_index(2:end), m_avg_estimate_risk(i, :), 'm', 'LineWidth', 1);
    title(title_label(i)); 
    legend(legend_label);
    xlabel(x_label); ylabel(y_label); xlim([0 max(v_time_index)]);
    grid on; grid minor;
end

% Plot the risk 
figure;
plot(v_time_index(2:end), v_avg_risk_sub, 'r', v_time_index(2:end), v_avg_risk_opt, 'b', ...
     v_time_index(2:end), v_avg_risk_risk, 'm', 'LineWidth', 1);
title('Risk');
legend({'Suboptimal PF', 'Optimal PF', 'Risk-aware MMSE'})
xlabel('Time [min]'); xlim([0 max(v_time_index)]);
ylabel('E[ V_y (||X - X_{est}||^2) ]');
grid on; grid minor;

% Plot MSE 
figure;
plot(v_time_index(2:end), v_avg_mse_sub , 'r', v_time_index(2:end), v_avg_mse_opt, 'b',...
     v_time_index(2:end), v_avg_mse_risk, 'm', 'LineWidth', 1);
title('MSE');
legend({'Suboptimal PF', 'Optimal PF', 'Risk-aware MMSE'})
xlabel('Time [min]'); xlim([0 max(v_time_index)]);
ylabel('MSE'); 
grid on; grid minor;