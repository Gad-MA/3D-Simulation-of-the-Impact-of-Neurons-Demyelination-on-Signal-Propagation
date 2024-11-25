import numpy as np
import matplotlib.pyplot as plt


# Dmylination effect parameters
isSensoryMylinated = 1
isExtensorMylinated = 1
isFlexorMylinated = 1
isInhibitorMylinated = 1


# Helper functions for gating variables
def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))


def beta_m(V):
    return 4.0 * np.exp(-(V + 65) / 18)


def alpha_h(V):
    return 0.07 * np.exp(-(V + 65) / 20)


def beta_h(V):
    return 1 / (1 + np.exp(-(V + 35) / 10))


def alpha_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))


def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)


class Neuron:
    """
    Represents a single neuron with Hodgkin-Huxley dynamics.
    """

    def __init__(self, Cm, g_Na, g_K, g_L, E_Na, E_K, E_L, I_ext=0.0):
        self.Cm = Cm  # Membrane capacitance (uF/cm^2)
        self.g_Na = g_Na  # Sodium conductance (mS/cm^2)
        self.g_K = g_K  # Potassium conductance (mS/cm^2)
        self.g_L = g_L  # Leak conductance (mS/cm^2)
        self.E_Na = E_Na  # Sodium reversal potential (mV)
        self.E_K = E_K  # Potassium reversal potential (mV)
        self.E_L = E_L  # Leak reversal potential (mV)
        self.I_ext = I_ext  # External current (uA/cm^2)

        # State variables
        self.V = -65.0  # Membrane potential (mV)
        self.m = alpha_m(self.V) / (
            alpha_m(self.V) + beta_m(self.V)
        )  # Sodium activation
        self.h = alpha_h(self.V) / (
            alpha_h(self.V) + beta_h(self.V)
        )  # Sodium inactivation
        self.n = alpha_n(self.V) / (
            alpha_n(self.V) + beta_n(self.V)
        )  # Potassium activation

    def update(self, dt, synaptic_input=0.0):
        """
        Updates the state of the neuron for one time step.

        Parameters:
            dt (float): Time step (ms)
            synaptic_input (float): Input current from synaptic connections (uA/cm^2)
        """

        # Total input current
        I_total = self.I_ext + synaptic_input

        # Compute ionic currents
        I_Na = self.g_Na * self.m**3 * self.h * (self.V - self.E_Na)
        I_K = self.g_K * self.n**4 * (self.V - self.E_K)
        I_L = self.g_L * (self.V - self.E_L)
        I_ion = I_total - (I_Na + I_K + I_L)

        # Update membrane potential
        self.V += (I_ion / self.Cm) * dt

        # Update gating variables
        self.m += (alpha_m(self.V) * (1 - self.m) - beta_m(self.V) * self.m) * dt
        self.h += (alpha_h(self.V) * (1 - self.h) - beta_h(self.V) * self.h) * dt
        self.n += (alpha_n(self.V) * (1 - self.n) - beta_n(self.V) * self.n) * dt


# Define parameters for the neurons in the patellar reflex circuit
Cm_normal = 1.0  # Normal membrane capacitance (uF/cm^2)
Cm_demyelinated = 5.0  # Increased capacitance for demyelinated neurons
g_Na, g_K, g_L = 120.0, 36.0, 0.3  # Conductances (mS/cm^2)
E_Na, E_K, E_L = 50.0, -77.0, -54.387  # Reversal potentials (mV)

# Create the neurons
sensory = Neuron(
    Cm_normal if isSensoryMylinated else Cm_demyelinated,
    g_Na,
    g_K,
    g_L,
    E_Na,
    E_K,
    E_L,
)  # Sensory neuron
extensor = Neuron(
    Cm_normal if isExtensorMylinated else Cm_demyelinated,
    g_Na,
    g_K,
    g_L,
    E_Na,
    E_K,
    E_L,
)  # Extensor motor neuron
inhibitory = Neuron(
    Cm_normal if isInhibitorMylinated else Cm_demyelinated,
    g_Na,
    g_K,
    g_L,
    E_Na,
    E_K,
    E_L,
)  # Inhibitory interneuron
flexor = Neuron(
    Cm_normal if isFlexorMylinated else Cm_demyelinated,
    g_Na,
    g_K,
    g_L,
    E_Na,
    E_K,
    E_L,
)  # Flexor motor neuron

# Simulation parameters
dt = 0.01  # Time step (ms)
T = 100  # Total simulation time (ms)
time = np.arange(0, T, dt)

# Initialize activity traces
sensory_trace, extensor_trace, inhibitory_trace, flexor_trace = [], [], [], []

# Simulation loop for the circuit
for t in time:
    # Synaptic connections
    synapse_sensory_extensor = max(0, sensory.V - 20)  # Excitatory to extensor
    synapse_sensory_inhibitory = max(0, sensory.V - 20)  # Excitatory to inhibitory
    synapse_inhibitory_flexor = -max(0, inhibitory.V - 20)  # Inhibitory to flexor

    # Update neurons
    sensory.update(dt, 10 if (0 <= t <= 1) else 0)
    extensor.update(dt, synapse_sensory_extensor)
    inhibitory.update(dt, synapse_sensory_inhibitory)
    flexor.update(dt, synapse_inhibitory_flexor)

    # Record activity
    sensory_trace.append(sensory.V)
    extensor_trace.append(extensor.V)
    inhibitory_trace.append(inhibitory.V)
    flexor_trace.append(flexor.V)

"""
Plot results overlapping
"""
plt.figure(figsize=(12, 8))
plt.plot(time, sensory_trace, "b", label="Sensory (afferent) neuron")
plt.plot(time, extensor_trace, "g", label="Extensor Neuron")
plt.plot(time, inhibitory_trace, "r", label="Inhibitory Neuron")
plt.plot(time, flexor_trace, "purple", label="Flexor Neuron")
plt.title("Patellar Reflex Circuit: Normal Neurons")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.grid()

"""
Plotting separated
"""
# Plot results
plt.figure(figsize=(12, 10))
# Sensory neuron
plt.subplot(4, 1, 1)
plt.plot(time, sensory_trace, "b", label="Sensory (afferent) neuron")
plt.ylabel("Voltage (mV)")
plt.legend()
plt.grid()
# Extensor motor neuron
plt.subplot(4, 1, 2)
plt.plot(time, extensor_trace, "g", label="Extensor Neuron")
plt.ylabel("Voltage (mV)")
plt.legend()
plt.grid()
# Inhibitory interneuron
plt.subplot(4, 1, 3)
plt.plot(time, inhibitory_trace, "r", label="Inhibitory Neuron")
plt.ylabel("Voltage (mV)")
plt.legend()
plt.grid()
# Flexor motor neuron
plt.subplot(4, 1, 4)
plt.plot(time, flexor_trace, "purple", label="Flexor Neuron")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.grid()
plt.legend()

plt.show()
