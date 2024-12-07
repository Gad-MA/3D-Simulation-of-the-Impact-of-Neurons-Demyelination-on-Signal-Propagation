import numpy as np
from HH_Computational_Model import computional_model, I_stim, dt
import matplotlib.pyplot as plt

stimulus_initial_time = 10
stimulus_duration = 1
simulation_duration = 50
model = computional_model(
    stimulus_initial_time, stimulus_duration, simulation_duration, 1, 1, 1, 1
)

"""
Plotting separated
"""
t = np.arange(0, simulation_duration, dt)
# Plot results
plt.figure(figsize=(12, 10))

# Stimulus
plt.subplot(5, 1, 1)
plt.plot(
    t,
    [I_stim(ti, stimulus_initial_time, stimulus_duration) for ti in t],
    "k",
    label="Stimulus",
)
plt.ylabel("Current\n(μA/cm²)")
plt.title("Knee-jerk Reflex Simulation")
plt.legend()

# Sensory neuron
plt.subplot(5, 1, 2)
plt.plot(t, model["sensory"], "b", label="Sensory")
plt.ylabel("Voltage (mV)")
plt.legend()

# Extensor motor neuron
plt.subplot(5, 1, 3)
plt.plot(t, model["extensor"], "g", label="Extensor")
plt.ylabel("Voltage (mV)")
plt.legend()

# Inhibitory interneuron
plt.subplot(5, 1, 4)
plt.plot(t, model["inhibitory"], "r", label="Inhibitory")
plt.ylabel("Voltage (mV)")
plt.legend()

# Flexor motor neuron
plt.subplot(5, 1, 5)
plt.plot(t, model["flexor"], "purple", label="Flexor")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.legend()

"""
Plotting overlapping
"""
# Plot results
plt.figure(figsize=(12, 10))

# Stimulus
plt.subplot(2, 1, 1)
plt.plot(
    t,
    [I_stim(ti, stimulus_initial_time, stimulus_duration) for ti in t],
    "k",
    label="Stimulus",
)
plt.ylabel("Current\n(μA/cm²)")
plt.title("Knee-jerk Reflex Simulation")
plt.legend()

# Sensory neuron
plt.subplot(2, 1, 2)
plt.plot(t, model["sensory"], "b", label="Sensory")
plt.plot(t, model["extensor"], "g", label="Extensor")
plt.plot(t, model["inhibitory"], "r", label="Inhibitory")
plt.plot(t, model["flexor"], "purple", label="Flexor")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.legend()


plt.tight_layout()
plt.show()
