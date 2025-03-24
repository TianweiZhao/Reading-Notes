###########################
# import 
###########################

import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt



##############################################
# 1. System Dynamics: Non-linear CSTR model
##############################################
def cstr_dynamics(x, t, u, Tf, Caf):
    """
    Computes the time derivative of the state for the CSTR.
    
    State vector x = [Ca, Cb, Cc, T, V] where:
      Ca: concentration of reactant A (mol/m^3)
      Cb: concentration of product B (mol/m^3) 
      Cc: concentration of product C (mol/m^3)
      T:  reactor temperature (K)
      V:  reactor volume (m^3)
      
    Control vector u = [Tc, Fin] where:
      Tc: Cooling jacket temperature (K)
      Fin: Inlet flow rate (m^3/min)
      
    Tf is the feed temperature (K) and Caf is the feed concentration of A (mol/m^3).
    
    The function includes:
      - Material balances (mass balances) for chemical species.
      - Energy balance including heat generated/removed by reactions and cooling.
      - Volume balance.
    """
    # Unpack control inputs
    Tc = u[0]  # Cooling jacket temperature
    Fin = u[1] # Inlet flow rate

    # Unpack state variables
    Ca, Cb, Cc, T, V = x

    # Process parameters
    Tf = 350         # Feed temperature (K)
    Fout = 100       # Outlet flow rate (m^3/min)
    rho = 1000       # Density (kg/m^3)
    Cp = 0.239       # Heat capacity (J/kg-K)
    UA = 5e4         # Overall heat transfer coefficient * area (W/m^2-K)

    # Reaction A -> B parameters (Arrhenius kinetics)
    mdelH_AB = 5e3   # Heat of reaction (J/mol)
    EoverR_AB = 8750 # Activation energy over gas constant (K)
    k0_AB = 7.2e10   # Pre-exponential factor (1/min)
    rA = k0_AB * np.exp(-EoverR_AB / T) * Ca

    # Reaction B -> C parameters (Arrhenius kinetics)
    mdelH_BC = 4e3   # Heat of reaction (J/mol)
    EoverR_BC = 10750# Activation energy over gas constant (K)
    k0_BC = 8.2e10   # Pre-exponential factor (1/min)
    rB = k0_BC * np.exp(-EoverR_BC / T) * Cb

    # Material balances (mass derivatives)
    dCadt = (Fin * Caf - Fout * Ca) / V - rA
    dCbdt = rA - rB - (Fout * Cb / V)
    dCcdt = rB - (Fout * Cc / V)

    # Energy balance (temperature derivative)
    dTdt = (Fin / V) * (Tf - T) \
           + (mdelH_AB / (rho * Cp)) * rA \
           + (mdelH_BC / (rho * Cp)) * rB \
           + (UA / (V * rho * Cp)) * (Tc - T)

    # Volume balance (volume derivative)
    dVdt = Fin - Fout

    return np.array([dCadt, dCbdt, dCcdt, dTdt, dVdt])




##############################################
# 2. Vector-Form PID Controller
##############################################

def PID_velocity(Ks, e, e_history, u_prev, dt):
    """
    Computes the control update using two velocity-form PID controllers.
    PID controller 1 [Kp_Cb, Ki_Cb, Kd_Cb] controls the concentration of B
    PID controller 2 [Kp_V, Ki_V, Kd_V] controls the volume

    Inputs:
      Ks: array of PID gains [Kp_Cb, Ki_Cb, Kd_Cb, Kp_V, Ki_V, Kd_V]
      e: current error vector [e_Cb, e_V] where e = setpoint - measurement
      e_history: array of previous errors; must contain at least two previous error values
      u_prev: the previous control action (array: [Tc, Fin])
      dt: time step
      
    The velocity-form PID update calculates the change in control signal and adds it to the previous control.
    The control update is clamped within operational limits.
    """
    # Unpack PID gains for the Cb-loop and V-loop
    Kp_Cb = Ks[0]
    Ki_Cb = Ks[1] + 1e-8  # Avoid division by zero
    Kd_Cb = Ks[2] + 1e-8
    
    Kp_V  = Ks[3]
    Ki_V  = Ks[4] + 1e-8
    Kd_V  = Ks[5] + 1e-8
    
    # Calculate control update for cooling temperature (Tc) for the Cb loop
    # Using the velocity (delta) form of PID:
    delta_u_Cb = (Kp_Cb * (e[0] - e_history[-1, 0])
                  + (Kp_Cb / Ki_Cb) * e[0] * dt
                  - Kp_Cb * Kd_Cb * (e[0] - 2 * e_history[-1, 0] + e_history[-2, 0]) / dt)
    Tc = u_prev[-1][0] + delta_u_Cb
    # Clamp Tc within operational limits (e.g., 290 to 450 K)
    Tc = np.clip(Tc, 290, 450)
    
    # Calculate control update for inlet flow rate (Fin) for the V loop
    delta_u_V = (Kp_V * (e[1] - e_history[-1, 1])
                 + (Kp_V / Ki_V) * e[1] * dt
                 - Kp_V * Kd_V * (e[1] - 2 * e_history[-1, 1] + e_history[-2, 1]) / dt)
    Fin = u_prev[-1][1] + delta_u_V
    # Clamp Fin within operational limits (e.g., 95 to 105 m^3/min)
    Fin = np.clip(Fin, 95, 105)
    
    return np.array([Tc, Fin])



##############################################
# 3. CSTR Environment written in Gym style
##############################################

class CSTRRLEnv(gym.Env):
    """
    A Gym environment for the CSTR system with an embedded velocity PID controller.
    
    Added new features:
        1. Parameter Uncertainty
        2. Measurement Noise
        3. Actuator (Control) Delays
        4. Transport/Dead Time Delays
        5. Disturbances
        6. Visualization capability

    Observation:
      A vector including:
         - Current reactor state ([Cb, T, V])
         - Previous reactor state (for derivative estimation)
         - Setpoints for Cb and V
      
    Action:
      A 6-dimensional continuous vector (normalized in [-1, 1]) representing PID gains:
         [Kp_Cb, Ki_Cb, Kd_Cb, Kp_V, Ki_V, Kd_V]
      
    Reward:
      Negative squared error between the measured variables and setpoints.
    """
    def __init__(self, simulation_steps=100, dt=1.0, 
                 uncertainty_level=0.1, 
                 noise_level=0.02,
                 actuator_delay_steps=1,
                 transport_delay_steps=2,
                 enable_disturbances=True):
        super(CSTRRLEnv, self).__init__()

        # simulate parameters
        self.sim_steps = simulation_steps # number of steps per episode
        self.dt = dt                      # time step for integration

        # Uncertainty and noise parameters
        self.uncertainty_level = uncertainty_level # level of parameter uncertainty (fraction)
        self.noise_level = noise_level             # level of measurement noise (fraction)

        # Delay Parameters
        self.actuator_delay_steps = actuator_delay_steps     # Control signal delay
        self.transport_delay_steps = transport_delay_steps   # Transport/Dead time delay

        # Disturbance Parameters
        self.enable_disturbances = enable_disturbances
        self.disturbance_interval = 20        # Time setps between disturbances
        self.next_disturbances = self.disturbance_interval # Steps until next disturbance

        # Define action space: 6 PID gains normalized in [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)

        # Define observation space: [Cb, T, V, Cb_prev, T_prev, V_prev, Cb_setpoint, V_setpoint]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 300.0, 80.0, 0.0, 300.0, 80.0, 0.0, 80.0]),
            high=np.array([1.0, 400.0, 120.0, 1.0, 400.0, 120.0, 1.0, 120.0]),
            dtype=np.float64
        )

        # PID gain scaling: map normalized action to actual PID gains
        # For Cb loop: lower = [-5, 0, 0.02], upper = [25, 20, 10]
        # For V loop: lower = [0, 0, 0.01], upper = [1, 2, 1]
        self.pid_lower = np.array([-5, 0, 0.02, 0, 0, 0.01])
        self.pid_upper = np.array([25, 20, 10, 1, 2, 1])

        # Setpoints for the controlled variables (constant case)
        self.setpoint_Cb = 0.70  # desired concentration of B (mol/m^3)
        self.setpoint_V  = 100.0 # desired volume (m^3)

        # Initialize reator conditions
        # x = [Ca, Cb, Cc, T, V]
        self.x0 = np.array([0.8, 0.0, 0.0, 325.0, 100.0])

        # Generate uncertain parameters (once per environment instantiate)
        # These represents real process parameters that differ from model assumptions
        # Adjust this section for "reality"
        self.process_params = {
            'Tf': 350 * (1 + self.uncertainty_level * (np.random.rand() - 0.5)),  # Feed temperature
            'Caf': 1.0 * (1 + self.uncertainty_level * (np.random.rand() - 0.5)), # Feed concentration
            'UA': 5e4 * (1 + self.uncertainty_level * (np.random.rand() - 0.5)),  # Heat transfer coefficient
            'k0_AB': 7.2e10 * (1 + self.uncertainty_level * (np.random.rand() - 0.5)), # Reaction rate constant
            'k0_BC': 8.2e10 * (1 + self.uncertainty_level * (np.random.rand() - 0.5)), # Reaction rate constant
        }

        # Initialize histories for PID: use lists to store previous errors and control actions
        self.e_history = [] # each element will be a 2D error vector [e_Cb, e_V]
        self.u_history = [] # each element will be a 2D control action [Tc, Fin]

        # Buffer for delayed measurements and control actions
        self.measurement_buffer = [] # Buffer for delayed measurements
        self.control_buffer = []     # Buffer for delayed control actions

        # For integration, keep the current state and step constant
        self.state = self.x0.copy()
        self.true_state = self.x0.copy()
        self.current_step = 0

        # For visualization
        self.history = {
            'time': [],
            'Cb': [],
            'T': [],
            'V': [],
            'Tc': [],
            'Fin': [],
            'state': [],
            'setpoint_Cb': [],
            'setpoint_V': []
        }

    # -----------------------------------------
    # Define reset function
    # -----------------------------------------
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset state and variables
        self.state = self.x0.copy()
        self.true_state = self.x0.copy()
        self.current_step = 0

        # Reset disturbance timing
        self.next_disturbances = self.disturbance_interval

        # Update uncertain parameters for this episode
        self.process_params = {
            'Tf': 350 * (1 + self.uncertainty_level * (np.random.rand() - 0.5)),
            'Caf': 1.0 * (1 + self.uncertainty_level * (np.random.rand() - 0.5)),
            'UA': 5e4 * (1 + self.uncertainty_level * (np.random.rand() - 0.5)),
            'k0_AB': 7.2e10 * (1 + self.uncertainty_level * (np.random.rand() - 0.5)),
            'k0_BC': 8.2e10 * (1 + self.uncertainty_level * (np.random.rand() - 0.5)),
        }

        # Initialize control and error history with default values
        default_u = np.array([300.0, 100.0])
        self.u_history = [default_u, default_u] # two copies for derivative computation

        # Reset buffers for delayed measurements and control actions
        self.measurement_buffer = [self.apply_measurement_noise(self.state)] * max(1, self.transport_delay_steps)
        self.control_buffer = [default_u] * max(1, self.actuator_delay_steps)

        # Compute intial error using possibly delayed and noisy measurements
        measured_state = self.measurement_buffer[0]
        initial_error = np.array([self.setpoint_Cb - measured_state[1],
                                  self.setpoint_V - measured_state[4]])
        
        # Initialize error history with two copies
        self.e_history = [initial_error, initial_error]


        # Reset history for visualization
        self.history = {
            'time': [0],
            'Cb': [self.state[1]],
            'T': [self.state[3]],
            'V': [self.state[4]],
            'Tc': [default_u[0]],
            'Fin': [default_u[1]],
            'setpoint_Cb': [self.setpoint_Cb],
            'setpoint_V': [self.setpoint_V]
        }

        # Create initial observation
        obs = np.array([
            measured_state[1], measured_state[3], measured_state[4], # current state
            measured_state[1], measured_state[3], measured_state[4], # previous state
            self.setpoint_Cb, self.setpoint_V # setpoints
        ], dtype=np.float64)

        return obs, {}
    
    # -----------------------------------------
    # Add noise to measurements
    # -----------------------------------------
    def apply_measurement_noise(self, state):
        """
        Add noise to the state measurements.
        """
        noisy_state = state.copy()

        # Add random noise scaled by the state values and noise level
        for i in range(len(state)):
            # Add relative noise (proportional to state value)
            noise = state[i] * self.noise_level * np.random.normal()
            # Ensure no negative concentrations or volumes
            noisy_state[i] = max(0, state[i] + noise)

        return noisy_state
    
    # -----------------------------------------
    # Apply disturbances
    # -----------------------------------------
    def apply_disturbances(self):
        """
        Apply a random disturbances to the system
        Types of disturbances:
        1. Step change in feed temperature
        2. Step change in feed concentration
        3. Brief cooling system upset
        """
        disturbance_type = np.random.randint(0, 3)

        if disturbance_type == 0:
            # Feed temperature step
            self.process_params['Tf'] *= (1 + 0.1 * (np.random.rand() - 0.5))
            return "Feed temperature disturbance"
        
        elif disturbance_type == 1:
            # Feed concentration step
            self.process_params['Caf'] *= (1 + 0.1 * (np.random.rand() - 0.5))
            return "Feed concentration disturbance"
        
        else: 
            # Brief cooling system upset (lasts 3 steps)
            self.process_params['UA'] *= 0.8 # Reduced heat transfer
            return "Cooling system upset"

    def custom_cstr_dynamics(self, x, t, u):
        """
        Custom CSTR dynamics with uncertain parameters and possible disturbances. 
        """
        # Unpack control inputs
        Tc = u[0]  # Cooling jacket temperature
        Fin = u[1] # Inlet flow rate

        # Unpack state variables
        Ca, Cb, Cc, T, V = x

        # Process parameters with uncertainty
        Tf = self.process_params['Tf']
        Caf = self.process_params['Caf']
        UA = self.process_params['UA']
        k0_AB = self.process_params['k0_AB']
        k0_BC = self.process_params['k0_BC']

        # Fixed parameters
        Fout = 100       # Outlet flow rate (m^3/min)
        rho = 1000       # Density (kg/m^3)
        Cp = 0.239       # Heat capacity (J/kg-K)

        # Reaction A -> B parameters (Arrhenius kinetics)
        mdelH_AB = 5e3   # Heat of reaction (J/mol)
        EoverR_AB = 8750 # Activation energy over gas constant (K)
        rA = k0_AB * np.exp(-EoverR_AB / T) * Ca

        # Reaction B -> C parameters (Arrhenius kinetics)
        mdelH_BC = 4e3    # Heat of reaction (J/mol)
        EoverR_BC = 10750 # Activation energy over gas constant (K)
        rB = k0_BC * np.exp(-EoverR_BC / T) * Cb

        # Material balances (mass derivatives)
        dCadt = (Fin * Caf - Fout * Ca) / V - rA
        dCbdt = rA - rB - (Fout * Cb / V)
        dCcdt = rB - (Fout * Cc / V)

        # Energy balance (temperature derivative)
        dTdt = (Fin / V) * (Tf - T) \
               + (mdelH_AB / (rho * Cp)) * rA \
               + (mdelH_BC / (rho * Cp)) * rB \
               + (UA / (V * rho * Cp)) * (Tc - T)
        
        # Volume balance (volume derivative)
        dVdt = Fin - Fout

        return np.array([dCadt, dCbdt, dCcdt, dTdt, dVdt])


    # -----------------------------------------
    # Define step function
    # -----------------------------------------
    def step(self, action):
        """
        Take one step in the simulation including:
        - Parameter uncertainty
        - Measurement noise
        - Actuator (control) delays
        - Transport/Dead time delays
        - Disturbances
        """
        # Scale normalized action to actual PID gains
        pid_gains = ((action + 1) / 2) * (self.pid_upper - self.pid_lower) + self.pid_lower

        # Get the current (possibly delayed and noisy) measurements
        measured_state = self.measurement_buffer[0]

        # Compute the error based on measurement
        current_error = np.array([self.setpoint_Cb - measured_state[1],
                                  self.setpoint_V - measured_state[4]])
        
        # Determine control action using velocity PID
        if self.current_step < 2:
            control_action = self.u_history[-1]
        else: 
            control_action = PID_velocity(pid_gains, current_error, self.e_history, self.u_history, self.dt)

        # Add new control action to buffer (introducing actuactor delay)
        self.control_buffer.append(control_action)
        # Get the delayed control action to apply
        delayed_control = self.control_buffer.pop(0)

        # Store the new control action and error in history
        self.u_history.append(control_action)
        self.e_history.append(current_error)

        # Apply disturbances if enabled
        disturbance_info = None
        if self.enable_disturbances:
            # Check if it's time for a disturbance
            if self.current_step >= self.next_disturbances:
                disturbance_info = self.apply_disturbances()
                self.next_disturbances += self.current_step + self.disturbance_interval

                # If it was a cooling system upset, schedule a fix after 3 steps
                if disturbance_info == "Cooling system upset":
                    # we'll restore the UA value after 3 steps
                    self.next_cooling_fix = self.current_step + 3


            # Fix cooling system (if needed)
            if hasattr(self, 'next_cooling_fix') and self.current_step == self.next_cooling_fix:
                # Restore UA to its original value
                self.process_params['UA'] /= 0.8  # Restore to original value
                delattr(self, 'next_cooling_fix') # Remove the attribute

        # Save current state before integration
        prev_state = self.true_state.copy()

        # Simulate the reactor dynamics using ODE integration with uncertain parameters
        t_span = [0, self.dt]
        new_state = odeint(self.custom_cstr_dynamics, self.true_state, t_span, args=(delayed_control,))[1]
        self.true_state = new_state.copy()

        # Apply measurement noise
        noisy_state= self.apply_measurement_noise(new_state)

        # Add new measurement to buffer (introducing measurement/transport delay)
        self.measurement_buffer.append(noisy_state)
        # Get the delayed measurement
        self.state = self.measurement_buffer.pop(0)

        # Compute reward: negative sum of squared errors (use true state for more accurate reward)
        true_error = np.array([self.setpoint_Cb - self.true_state[1],
                               self.setpoint_V - self.true_state[4]])
        reward = -np.sum(true_error ** 2)

        # Construct the observation with delayed, noisy measurements
        obs = np.array([
            self.state[1], self.state[3], self.state[4], # current state
            prev_state[1], prev_state[3], prev_state[4], # previous state
            self.setpoint_Cb, self.setpoint_V # setpoints
        ], dtype=np.float64)
        
        # Update history for visualization
        self.history['time'].append(self.current_step * self.dt)
        self.history['Cb'].append(self.true_state[1]) # Store true values for visualization
        self.history['T'].append(self.true_state[3])
        self.history['V'].append(self.true_state[4])
        self.history['Tc'].append(delayed_control[0])
        self.history['Fin'].append(delayed_control[1])
        self.history['setpoint_Cb'].append(self.setpoint_Cb)
        self.history['setpoint_V'].append(self.setpoint_V)

        self.current_step += 1
        done = self.current_step >= self.sim_steps

        # Into dict can include debugging information and disturbance info
        info = {
            "pid_gains": pid_gains,
            "control_action": control_action,
            "true_state": self.true_state,
            "disturbance": disturbance_info
        }

        return obs, reward, done, False, info


    # -----------------------------------------
    # Define render function
    # -----------------------------------------
    def render(self, mode='human'):
        """
        Render the environment's current state
        Shows plots of controlled variables (Cb, V) and temperature over time
        Also shows control actions (Tc, Fin)
        """
        if not hasattr(self, 'fig') or self.fig is None:
            # Create figure on the first call
            plt.ion() # Interactive mode
            self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle("CSTR Control System")

            # Configure subplots
            self.axs[0, 0].set_title("Product B Concentration (Cb)")
            self.axs[0, 0].set_xlabel("Time (min)")
            self.axs[0, 0].set_ylabel("Concentration B (mol/m^3)")

            self.axs[0, 1].set_title("Reactor Volume (V)")
            self.axs[0, 1].set_xlabel("Time (min)")
            self.axs[0, 1].set_ylabel("Volume (m^3)")

            self.axs[1, 0].set_title("Reactor Temperature (T)")
            self.axs[1, 0].set_xlabel("Time (min)")
            self.axs[1, 0].set_ylabel("Temperature (K)")

            self.axs[1, 1].set_title("Control Actions")
            self.axs[1, 1].set_xlabel("Time (min)")
            self.axs[1, 1].set_ylabel("Value")

            plt.tight_layout()

        # Update plots
        if self.current_step > 0:
            # Clear previous plots
            for ax in self.axs.flatten():
                ax.clear()

            # Plot Cb and setpoint
            self.axs[0, 0].plot(self.history['time'], self.history['Cb'], 'b-', label='Actual Cb')
            self.axs[0, 0].plot(self.history['time'], self.history['setpoint_Cb'], 'r--', label='Setpoint')
            self.axs[0, 0].legend()
            self.axs[0, 0].set_title('Product B Concentration')
            self.axs[0, 0].set_xlabel('Time (min)')
            self.axs[0, 0].set_ylabel('Cb (mol/m³)')

            # Plot Volume and setpoint
            self.axs[0, 1].plot(self.history['time'], self.history['V'], 'g-', label='Actual Volume')
            self.axs[0, 1].plot(self.history['time'], self.history['setpoint_V'], 'r--', label='Setpoint')
            self.axs[0, 1].legend()
            self.axs[0, 1].set_title('Reactor Volume')
            self.axs[0, 1].set_xlabel('Time (min)')
            self.axs[0, 1].set_ylabel('Volume (m³)')

            # Plot Temperature
            self.axs[1, 0].plot(self.history['time'], self.history['T'], 'r-', label='Reactor Temp')
            self.axs[1, 0].legend()
            self.axs[1, 0].set_title('Reactor Temperature')
            self.axs[1, 0].set_xlabel('Time (min)')
            self.axs[1, 0].set_ylabel('Temperature (K)')

            # Plot Control Actions
            self.axs[1, 1].plot(self.history['time'], self.history['Tc'], 'b-', label='Cooling Temp')
            self.axs[1, 1].plot(self.history['time'], self.history['Fin'], 'g-', label='Inlet Flow')
            self.axs[1, 1].legend()
            self.axs[1, 1].set_title('Control Actions')
            self.axs[1, 1].set_xlabel('Time (min)')
            self.axs[1, 1].set_ylabel('Value')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

        return self.fig 
    

    # -----------------------------------------
    # Define close function
    # -----------------------------------------
    def close(self):
        """
        Close the environment and clean up resources.
        """
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            plt.ioff()



            