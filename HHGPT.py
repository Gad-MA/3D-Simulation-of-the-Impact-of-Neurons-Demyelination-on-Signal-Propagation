import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# Define constants as before
gK = 36.0; gNa = 120.0; gL = 0.3; Cm = 1.0
VK = -12.0; VNa = 115.0; Vl = 10.613
tmin, tmax = 0.0, 50.0
T = np.linspace(tmin, tmax, 10000)

class Neuron:
    def __init__(self, g_syn=0.1, V_syn=0.0):
        self.Vm = 0.0
        self.n = self.n_inf()
        self.m = self.m_inf()
        self.h = self.h_inf()
        self.g_syn = g_syn      # Synaptic conductance
        self.V_syn = V_syn      # Synaptic reversal potential
        self.s = 0.0            # Synaptic activation variable
        self.alpha = 1.0        # Activation rate
        self.beta = 0.5         # Deactivation rate
    
    def alpha_n(self, Vm): return (0.01 * (10.0 - Vm)) / (np.exp(1.0 - (0.1 * Vm)) - 1.0)
    def beta_n(self, Vm): return 0.125 * np.exp(-Vm / 80.0)
    def alpha_m(self, Vm): return (0.1 * (25.0 - Vm)) / (np.exp(2.5 - (0.1 * Vm)) - 1.0)
    def beta_m(self, Vm): return 4.0 * np.exp(-Vm / 18.0)
    def alpha_h(self, Vm): return 0.07 * np.exp(-Vm / 20.0)
    def beta_h(self, Vm): return 1.0 / (np.exp(3.0 - (0.1 * Vm)) + 1.0)
    
    def n_inf(self): return self.alpha_n(self.Vm) / (self.alpha_n(self.Vm) + self.beta_n(self.Vm))
    def m_inf(self): return self.alpha_m(self.Vm) / (self.alpha_m(self.Vm) + self.beta_m(self.Vm))
    def h_inf(self): return self.alpha_h(self.Vm) / (self.alpha_h(self.Vm) + self.beta_h(self.Vm))
    
    # Synaptic current update
    def update_synaptic_activation(self, pre_spike):
        if pre_spike:
            self.s = self.alpha * (1 - self.s) - self.beta * self.s
    
    def compute_derivatives(self, y, t, I_ext=0):
        Vm, n, m, h = y
        GK = (gK / Cm) * np.power(n, 4.0)
        GNa = (gNa / Cm) * np.power(m, 3.0) * h
        GL = gL / Cm

        I_syn = self.g_syn * self.s * (Vm - self.V_syn)
        
        dVm_dt = (I_ext / Cm) - (GK * (Vm - VK)) - (GNa * (Vm - VNa)) - (GL * (Vm - Vl)) - I_syn
        dn_dt = self.alpha_n(Vm) * (1.0 - n) - self.beta_n(Vm) * n
        dm_dt = self.alpha_m(Vm) * (1.0 - m) - self.beta_m(Vm) * m
        dh_dt = self.alpha_h(Vm) * (1.0 - h) - self.beta_h(Vm) * h
        
        return [dVm_dt, dn_dt, dm_dt, dh_dt]

# Initialize neurons with appropriate synaptic conductance and reversal potentials
sensory_neuron = Neuron(g_syn=0.1, V_syn=0.0)     # Excitatory connection to interneuron
interneuron = Neuron(g_syn=0.05, V_syn=-70.0)     # Inhibitory connection to motor neuron
motor_neuron = Neuron(g_syn=0.1, V_syn=0.0)       # Excitatory

# Initialize states for each neuron (Vm, n, m, h)
Y_sensory = [0.0, sensory_neuron.n_inf(), sensory_neuron.m_inf(), sensory_neuron.h_inf()]
Y_interneuron = [0.0, interneuron.n_inf(), interneuron.m_inf(), interneuron.h_inf()]
Y_motor = [0.0, motor_neuron.n_inf(), motor_neuron.m_inf(), motor_neuron.h_inf()]

# Input stimulus function
def sensory_input(t):
    return 150.0 if 10.0 < t < 11.0 else 0.0

# Storage for results
Vm_sensory = []
Vm_interneuron = []
Vm_motor = []

# Simulation loop
for i, t in enumerate(T):
    # Sensory neuron stimulus
    I_sensory = sensory_input(t)
    
    # Update each neuron
    Y_sensory = odeint(sensory_neuron.compute_derivatives, Y_sensory, [t], args=(I_sensory,))[0]
    Y_interneuron = odeint(interneuron.compute_derivatives, Y_interneuron, [t], args=(0,))[0]
    Y_motor = odeint(motor_neuron.compute_derivatives, Y_motor, [t], args=(0,))[0]
    
    # Check if sensory neuron spiked (threshold of 20 mV) and activate synapse
    if Y_sensory[0] > 20.0:
        interneuron.update_synaptic_activation(True)
    if Y_interneuron[0] > 20.0:
        motor_neuron.update_synaptic_activation(True)

    # Store results
    Vm_sensory.append(Y_sensory[0])
    Vm_interneuron.append(Y_interneuron[0])
    Vm_motor.append(Y_motor[0])

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(T, Vm_sensory, label='Sensory Neuron Vm')
plt.plot(T, Vm_interneuron, label='Interneuron Vm')
plt.plot(T, Vm_motor, label='Motor Neuron Vm')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.title("Knee-Jerk Reflex Simulation with Hodgkin-Huxley Neurons")
plt.show()
