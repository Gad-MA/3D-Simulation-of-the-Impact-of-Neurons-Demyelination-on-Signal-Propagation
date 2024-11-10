import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import csv

# Dmylination effect parameters
isSensoryMylinated = 1
isExtensorMylinated = 1
isFlexorMylinated = 1
isInhibitorMylinated = 1

Cm_demylinated = 5

# Hodgkin-Huxley Parameters
Cm = 1
g_Na, g_K, g_L = 120, 36, 0.3
E_Na, E_K, E_L = 60, -88, -54.387


# Gate dynamics
def alpha_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))


def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)


def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))


def beta_m(V):
    return 4 * np.exp(-(V + 65) / 18)


def alpha_h(V):
    return 0.07 * np.exp(-(V + 65) / 20)


def beta_h(V):
    return 1 / (1 + np.exp(-(V + 35) / 10))


# Ionic currents
def I_Na(V, m, h):
    return g_Na * m**3 * h * (V - E_Na)


def I_K(V, n):
    return g_K * n**4 * (V - E_K)


def I_L(V):
    return g_L * (V - E_L)


# Synaptic parameters
g_syn_ex = 0.5  # excitatory synaptic conductance
g_syn_in = 1.0  # inhibitory synaptic conductance
E_syn_ex = 0  # excitatory reversal potential
E_syn_in = -80  # inhibitory reversal potential
tau_syn = 2.0  # synaptic time constant


# Stimulus function (tap to the knee)
def I_stim(t):
    return 40 * (t > 10) * (t < 11)  # Brief 1ms stimulus at t=10ms


def synaptic_current(V_pre, V_post, g_syn, E_syn):
    # Simple threshold-based synaptic transmission
    threshold = -20
    return g_syn * (V_pre > threshold) * (E_syn - V_post)


# Combined dynamics for all 4 neurons
def dSystem_dt(X, t):
    # Unpack state variables for each neuron
    # Each neuron has 4 variables: V, m, h, n
    V_sens, m_sens, h_sens, n_sens = X[0:4]  # Sensory neuron
    V_ext, m_ext, h_ext, n_ext = X[4:8]  # Extensor motor neuron
    V_inh, m_inh, h_inh, n_inh = X[8:12]  # Inhibitory interneuron
    V_flex, m_flex, h_flex, n_flex = X[12:16]  # Flexor motor neuron

    # Calculate synaptic currents
    I_sens_ext = synaptic_current(
        V_sens, V_ext, g_syn_ex, E_syn_ex
    )  # Sensory → Extensor
    I_sens_inh = synaptic_current(
        V_sens, V_inh, g_syn_ex, E_syn_ex
    )  # Sensory → Inhibitory
    I_inh_flex = synaptic_current(
        V_inh, V_flex, g_syn_in, E_syn_in
    )  # Inhibitory → Flexor

    # Sensory neuron dynamics
    dV_sens = (
        I_stim(t) - I_Na(V_sens, m_sens, h_sens) - I_K(V_sens, n_sens) - I_L(V_sens)
    ) / (Cm if isSensoryMylinated else Cm_demylinated)
    dm_sens = alpha_m(V_sens) * (1 - m_sens) - beta_m(V_sens) * m_sens
    dh_sens = alpha_h(V_sens) * (1 - h_sens) - beta_h(V_sens) * h_sens
    dn_sens = alpha_n(V_sens) * (1 - n_sens) - beta_n(V_sens) * n_sens

    # Extensor motor neuron dynamics
    dV_ext = (
        I_sens_ext - I_Na(V_ext, m_ext, h_ext) - I_K(V_ext, n_ext) - I_L(V_ext)
    ) / (Cm if isExtensorMylinated else Cm_demylinated)
    dm_ext = alpha_m(V_ext) * (1 - m_ext) - beta_m(V_ext) * m_ext
    dh_ext = alpha_h(V_ext) * (1 - h_ext) - beta_h(V_ext) * h_ext
    dn_ext = alpha_n(V_ext) * (1 - n_ext) - beta_n(V_ext) * n_ext

    # Inhibitory interneuron dynamics
    dV_inh = (
        I_sens_inh - I_Na(V_inh, m_inh, h_inh) - I_K(V_inh, n_inh) - I_L(V_inh)
    ) / (Cm if isInhibitorMylinated else Cm_demylinated)
    dm_inh = alpha_m(V_inh) * (1 - m_inh) - beta_m(V_inh) * m_inh
    dh_inh = alpha_h(V_inh) * (1 - h_inh) - beta_h(V_inh) * h_inh
    dn_inh = alpha_n(V_inh) * (1 - n_inh) - beta_n(V_inh) * n_inh

    # Flexor motor neuron dynamics
    dV_flex = (
        I_inh_flex - I_Na(V_flex, m_flex, h_flex) - I_K(V_flex, n_flex) - I_L(V_flex)
    ) / (Cm if isFlexorMylinated else Cm_demylinated)
    dm_flex = alpha_m(V_flex) * (1 - m_flex) - beta_m(V_flex) * m_flex
    dh_flex = alpha_h(V_flex) * (1 - h_flex) - beta_h(V_flex) * h_flex
    dn_flex = alpha_n(V_flex) * (1 - n_flex) - beta_n(V_flex) * n_flex

    return [
        dV_sens,
        dm_sens,
        dh_sens,
        dn_sens,
        dV_ext,
        dm_ext,
        dh_ext,
        dn_ext,
        dV_inh,
        dm_inh,
        dh_inh,
        dn_inh,
        dV_flex,
        dm_flex,
        dh_flex,
        dn_flex,
    ]


# Time vector
t = np.arange(0, 50, 0.01)  # 50ms simulation

# Initial conditions for all neurons (resting state)
X0 = np.array([-65, 0.05, 0.6, 0.32] * 4)  # Repeated for all 4 neurons

# Solve the system
X = odeint(dSystem_dt, X0, t)


def toCSV(l1, l2, l3, l4):
    with open("output.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([l1, l2, l3, l4])


sensoryNeuronActivation = X[:, 0]
extensorNeuronActivation = X[:, 4]
inhibitorNeuronActivation = X[:, 8]
flexorNeuronActivation = X[:, 12]

threshold = -55  # in mV

for i in range(5000):
    sensoryNeuronActivation[i] = 1 if (sensoryNeuronActivation[i] >= threshold) else 0
    extensorNeuronActivation[i] = 1 if (extensorNeuronActivation[i] >= threshold) else 0
    inhibitorNeuronActivation[i] = (
        1 if (inhibitorNeuronActivation[i] >= threshold) else 0
    )
    flexorNeuronActivation[i] = 1 if (flexorNeuronActivation[i] >= threshold) else 0

# print(sensoryNeuronActivation)
# print(extensorNeuronActivation)
# print(inhibitorNeuronActivation)
# print(flexorNeuronActivation)

toCSV(
    sensoryNeuronActivation,
    extensorNeuronActivation,
    inhibitorNeuronActivation,
    flexorNeuronActivation,
)
