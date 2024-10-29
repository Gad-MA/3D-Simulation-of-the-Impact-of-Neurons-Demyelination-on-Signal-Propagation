import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Hodgkin-Huxley parameters
class HHParams:
    def __init__(self):
        self.Cm = 1
        self.g_Na, self.g_K, self.g_L = 120, 36, 0.3
        self.E_Na, self.E_K, self.E_L = 60, -88, -54.387


# Hodgkin-Huxley channel kinetics
class HHKinetics:
    @staticmethod
    def alpha_n(V):
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

    @staticmethod
    def beta_n(V):
        return 0.125 * np.exp(-(V + 65) / 80)

    @staticmethod
    def alpha_m(V):
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

    @staticmethod
    def beta_m(V):
        return 4 * np.exp(-(V + 65) / 18)

    @staticmethod
    def alpha_h(V):
        return 0.07 * np.exp(-(V + 65) / 20)

    @staticmethod
    def beta_h(V):
        return 1 / (1 + np.exp(-(V + 35) / 10))


class Neuron:
    def __init__(self, params):
        self.params = params
        # Initial conditions: [V, m, h, n]
        self.state = np.array([-65, 0.05, 0.6, 0.32])

    def ionic_currents(self, V, m, h, n):
        I_Na = self.params.g_Na * m**3 * h * (V - self.params.E_Na)
        I_K = self.params.g_K * n**4 * (V - self.params.E_K)
        I_L = self.params.g_L * (V - self.params.E_L)
        return I_Na, I_K, I_L


class KneeJerkCircuit:
    def __init__(self):
        self.params = HHParams()
        self.kinetics = HHKinetics()

        # Create neurons
        self.sensory = Neuron(self.params)
        self.extensor = Neuron(self.params)
        self.inhibitory = Neuron(self.params)
        self.flexor = Neuron(self.params)

        # Synaptic weights
        self.w_sensory_extensor = 15  # Excitatory
        self.w_sensory_inhibitory = 10  # Excitatory
        self.w_inhibitory_flexor = -12  # Inhibitory

        # Synaptic delay (ms)
        self.delay = 1

    def simulate(self, t_span, dt):
        t = np.arange(0, t_span, dt)
        n_steps = len(t)

        # Arrays to store results
        V_sensory = np.zeros(n_steps)
        V_extensor = np.zeros(n_steps)
        V_inhibitory = np.zeros(n_steps)
        V_flexor = np.zeros(n_steps)

        # Initial states
        states = {
            "sensory": self.sensory.state.copy(),
            "extensor": self.extensor.state.copy(),
            "inhibitory": self.inhibitory.state.copy(),
            "flexor": self.flexor.state.copy(),
        }

        def I_stim(t):
            # Stimulus to sensory neuron (stretch receptor)
            return 40 if 100 <= t <= 110 else 0

        def dstatedt(X, t, neuron_type, I_syn=0):
            V, m, h, n = X

            # Calculate ionic currents
            I_Na, I_K, I_L = getattr(self, neuron_type).ionic_currents(V, m, h, n)

            # Membrane potential
            dVdt = (
                I_stim(t) if neuron_type == "sensory" else 0 + I_syn - I_Na - I_K - I_L
            ) / self.params.Cm

            # Channel gating variables
            dmdt = self.kinetics.alpha_m(V) * (1 - m) - self.kinetics.beta_m(V) * m
            dhdt = self.kinetics.alpha_h(V) * (1 - h) - self.kinetics.beta_h(V) * h
            dndt = self.kinetics.alpha_n(V) * (1 - n) - self.kinetics.beta_n(V) * n

            return [dVdt, dmdt, dhdt, dndt]

        # Simulation loop
        delay_steps = int(self.delay / dt)

        for i in range(n_steps):
            # Store current membrane potentials
            V_sensory[i] = states["sensory"][0]
            V_extensor[i] = states["extensor"][0]
            V_inhibitory[i] = states["inhibitory"][0]
            V_flexor[i] = states["flexor"][0]

            # Calculate synaptic currents with delay
            if i >= delay_steps:
                I_syn_extensor = self.w_sensory_extensor * (
                    V_sensory[i - delay_steps] > 0
                )
                I_syn_inhibitory = self.w_sensory_inhibitory * (
                    V_sensory[i - delay_steps] > 0
                )
                I_syn_flexor = self.w_inhibitory_flexor * (
                    V_inhibitory[i - delay_steps] > 0
                )
            else:
                I_syn_extensor = I_syn_inhibitory = I_syn_flexor = 0

            # Update each neuron's state
            dt_small = dt / 10  # Smaller integration steps for stability
            for _ in range(10):
                states["sensory"] += (
                    np.array(dstatedt(states["sensory"], t[i], "sensory")) * dt_small
                )
                states["extensor"] += (
                    np.array(
                        dstatedt(states["extensor"], t[i], "extensor", I_syn_extensor)
                    )
                    * dt_small
                )
                states["inhibitory"] += (
                    np.array(
                        dstatedt(
                            states["inhibitory"], t[i], "inhibitory", I_syn_inhibitory
                        )
                    )
                    * dt_small
                )
                states["flexor"] += (
                    np.array(dstatedt(states["flexor"], t[i], "flexor", I_syn_flexor))
                    * dt_small
                )

        return t, V_sensory, V_extensor, V_inhibitory, V_flexor


# Run simulation
circuit = KneeJerkCircuit()
t, V_sensory, V_extensor, V_inhibitory, V_flexor = circuit.simulate(300, 0.1)

# Plotting
plt.figure(figsize=(12, 10))

plt.subplot(1, 1, 1)
plt.plot(t, V_sensory, "b-", label="Sensory Neuron")
plt.ylabel("Voltage (mV)")
plt.title("Knee-jerk Reflex Circuit Simulation")
# plt.legend()

# plt.subplot(4, 1, 2)
plt.plot(t, V_extensor, "g-", label="Extensor Motor Neuron")
# plt.ylabel('Voltage (mV)')
# plt.legend()

# plt.subplot(4, 1, 3)
plt.plot(t, V_inhibitory, "r-", label="Inhibitory Interneuron")
# plt.ylabel('Voltage (mV)')
# plt.legend()

# plt.subplot(4, 1, 4)
plt.plot(t, V_flexor, "purple", label="Flexor Motor Neuron")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.legend()

plt.tight_layout()
plt.show()
