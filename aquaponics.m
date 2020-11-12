classdef aquaponics
    % Aquaponics class
    properties         
        % Parameters
        
        % Tank 1 = Fish tank
        % Tank 2 = Reactor tank
        
        volume_tank1;   % [l]
        volume_tank2;   % [l]
        
        flow_exc;       % [l / min]
        flow_tanks;     % [l / min]

        nhn_tank1;      % [mg / l]
        nhn_tank2;      % [mg / l]

        no3n_tank1;     % [mg / l]
        no3n_tank2;     % [mg / l]
        
        dot_nhn_tank1;  % derivative of nhn_tank1
        dot_nhn_tank2;  % derivative of nhn_tank2

        dot_no3n_tank1; % derivative of no3n_tank1
        dot_no3n_tank2; % derivative of no3n_tank2
        
        rel_filling;
        surface;        % [m^2 m^(-3)]
        total_surface;  % [m^2]
        
        nhx_conv;       % [g m^(-2) day^(-1)]
        conv_rate;      % [g / day^(-1)] 

        % Noise parameters
        mu_system    = zeros(4, 1); % Mean for system noise
        cov_system   = eye(4);      % Covariance for system noise
        mu_obs       = zeros(2, 1); % Mean for observation noise
        cov_obs      = eye(2);      % Covariance for observation noise

        % System
        v_state      = zeros(4, 1); % [nhn_tank1 nhn_tank2 no3n_tank1 no3n_tank2]'
        v_dot_state  = zeros(4, 1); % derivative of v_state
        v_obs        = zeros(2, 1); % [nhn_tank1 no3n_tank1]'
        v_control    = zeros(2, 1); % []

        % History
        v_state_history     = [];
        v_obs_history       = [];

        % System matrices
        m_a = zeros(4);
        m_b = zeros(4, 2);
        m_c = [1 0 0 0; 
               0 0 1 0];
        v_conv = zeros(4, 1); % Total mass converted NHx -> NO3-N
    end 
   
    methods
        function self = aquaponics() 
            % Class constructor 
            
            % Default values (if no user input)
            if nargin == 0
                % System parameters
                volume_tank1 = 6000;
                volume_tank2 = 1300;
                
                flow_exc    = 480;
                flow_tanks  = 300;
                
                rel_filling = 0.6;
                surface     = 300;
                nhx_conv    = 1.2;
                
                % Noise parameters
                cov_system  = 0.02 * diag([1.5 1 300 350]);
                cov_obs     = 0.15 * diag([1.5 300]);
                
                % Initial state
                nhn_tank1   = 0.01;
                nhn_tank2   = 0.01;

                no3n_tank1  = 0.01;
                no3n_tank2  = 0.01;
                
                % Initial derivative states
                dot_nhn_tank1 = 0;
                dot_nhn_tank2 = 0;
                
                dot_no3n_tank1 = 0;
                dot_no3n_tank2 = 0;
                
            end
            
            % Assign values
            self.volume_tank1 = volume_tank1;
            self.volume_tank2 = volume_tank2;

            self.flow_exc   = flow_exc;
            self.flow_tanks = flow_tanks;
            
            self.dot_nhn_tank1 = dot_nhn_tank1;
            self.dot_nhn_tank2 = dot_nhn_tank2;
            
            self.dot_no3n_tank1 = dot_no3n_tank1;
            self.dot_no3n_tank2 = dot_no3n_tank2;

            self.nhn_tank1 = nhn_tank1;
            self.nhn_tank2 = nhn_tank2;

            self.no3n_tank1 = no3n_tank1;
            self.no3n_tank2 = no3n_tank2;
            
            self.rel_filling = rel_filling;
            self.surface = surface;
            self.total_surface = self.volume_tank2 * self.surface * self.rel_filling;
            
            self.nhx_conv = nhx_conv;
            self.conv_rate = self.surface * self.nhx_conv;

            self.cov_system = cov_system;
            self.cov_obs = cov_obs;
            
            self = f_set_system(self);
            
            self = f_update_matrices(self);

            self.v_control = zeros(2, 1);
        end

        function self = f_update_matrices(self)
            % Update system matrices 

            self.m_a    = [ (-(self.flow_tanks + self.flow_exc) / (self.volume_tank1))       (self.flow_tanks / self.volume_tank1)           0                                                                   0;
                            (self.flow_tanks / self.volume_tank2)                              (-(self.flow_tanks / self.volume_tank2))      0                                                                   0;
                            0                                                                  0                                             (-(self.flow_tanks + self.flow_exc) / (self.volume_tank1))          (self.flow_tanks / self.volume_tank1);
                            0                                                                  0                                             (self.flow_tanks / self.volume_tank2)                               (-(self.flow_tanks / self.volume_tank2))];

            self.m_b    = [ (1 / self.volume_tank2)  0   0                           0;
                            0                        0   (1 / self.volume_tank2)     0]';
        end

        function self = f_set_system(self)
            % Setup initial states
            self.v_state = [self.nhn_tank1 self.nhn_tank2 self.no3n_tank1 self.no3n_tank2]';
            self.v_obs   = [self.nhn_tank1 self.no3n_tank1]';

            self.v_state_history = [self.v_state_history self.v_state];
            self.v_obs_history   = [self.v_obs_history self.v_obs];
        end
        
        function self = f_update_control(self, time, feed, nhn)
            % Food to Nitrogen convertion
            n_exc   = feed * 0.16 * 0.75 * (0.5 - 0.17);
            nhn_exc = n_exc * (sin(2 * pi * time / 1440) + 1);
            
            nhn_hydro = nhn * (sin(2 * pi * time / 720) + 1);
            
            % Update control input
            self.v_control = [nhn_exc   self.flow_exc * (nhn_hydro)]';
        end

        function self = f_update_noise(self, cov_system, cov_obs)
            % Update covariance matrices for system noise
            self.cov_system  = cov_system;
            self.cov_obs     = cov_obs;
        end

        function self = f_update_dynamics(self, time_delta, time, feed, nhn_hyd)
            % Set control values
            self = f_update_control(self, time, feed, nhn_hyd);
            
            % Update non-linear dynamics
            self.v_conv = (1 / self.volume_tank2) * [ 0    (-(2 * self.conv_rate * self.v_state(2)) / (self.conv_rate + 2 * self.v_state(2)))    0   ((2 * self.conv_rate * self.v_state(2)) / (self.conv_rate + 2 * self.v_state(2)))]';
            
            % Update noise
            v_state_noise = mvnrnd(self.mu_system, self.cov_system)'; 
            v_obs_noise   = mvnrnd(self.mu_obs, self.cov_obs)'; 
            
            % System dynamics
            self.v_dot_state = self.m_a * self.v_state + self.m_b * self.v_control + self.v_conv + v_state_noise;
            
            % Compute state
            self.v_state = self.v_dot_state * time_delta + self.v_state;
            
            % Compute observations
            self.v_obs   = self.m_c * self.v_state + v_obs_noise;
            
            % Save history of values
            self.v_state_history = [self.v_state_history self.v_state];
            self.v_obs_history   = [self.v_obs_history self.v_obs];
        end
    end
end