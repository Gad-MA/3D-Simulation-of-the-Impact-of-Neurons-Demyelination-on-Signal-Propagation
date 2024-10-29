import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

Cm = 1
g_Na, g_K, g_L = 120, 36, 0.3
E_Na, E_K, E_L = 60, -88, -54.387


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


def I_Na(V, m, h):
    return g_Na * m**3 * h * (V - E_Na)


def I_K(V, n):
    return g_K * n**4 * (V - E_K)


def I_L(V):
    return g_L * (V - E_L)


t = np.arange(0, 450, 0.01)  # time in ms


def I_inj(t):
    # return 10 * (t > 100) - 10 * (t > 200) + 35 * (t > 300) - 35 * (t > 400)
    return 40*(t>100) - 40*(t>110)

def dALLdt(X, t):
    V, m, h, n = X

    dVdt = (I_inj(t) - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / Cm
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n

    return dVdt, dmdt, dhdt, dndt


X = odeint(dALLdt, [-65, 0.05, 0.6, 0.32], t)
V = X[:, 0]

"""
Plotting
"""
plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.title("Hodgkin-Huxley model simulation")
plt.plot(t, V, "k")
plt.ylabel("V (mV)")

plt.subplot(2, 1, 2)
plt.plot(t, I_inj(t), "k")
plt.xlabel("t (ms)")
plt.ylabel("$I_{inj}$ ($\\mu{A}/cm^2$)")
plt.ylim(-1, 50)

plt.show()
