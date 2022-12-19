import numpy as np
import matplotlib.pyplot as plt


PI = np.pi
G = 9.81
KM = 370.0
CM = 0.23

def square(x):
    return x * x


def omega(k):
    return np.sqrt(G * k * (1.0 + square(k / KM)))


def phillips(x, u_hs=0.6, u_tp=30.0, u_resolution=512, u_size=512):
    x -= 0.5
    n = x[:]
    mask = x >= u_resolution * 0.5
    n[mask] = x[mask] - u_resolution
    k = 2.0 * PI * n / u_size
    l_wind = 0.5 * square(u_hs)
    Omega = 2.0 * PI / u_tp
    kp = G * square(Omega / l_wind)
    c = omega(k) / k
    cp = omega(kp) / kp
    Lpm = np.exp(-1.25 * square(kp / k))
    gamma = 1.7
    sigma = 0.08 * (1.0 + 4.0 * Omega**-3.0)
    Gamma = np.exp(-square(np.sqrt(k / kp) - 1.0) / 2.0 * square(sigma))
    Jp = gamma**Gamma
    Fp = Lpm * Jp * np.exp(-Omega / np.sqrt(10.0) * (np.sqrt(k / kp) - 1.0))
    alphap = 0.006 * np.sqrt(Omega)
    Bl = 0.5 * alphap * cp / c * Fp
    z0 = 0.000037 * np.square(l_wind) / G * (l_wind / cp)**0.9
    uStar = 0.41 * l_wind / np.log(10.0 / z0)
    alpham = 0.01 * ((1.0 + np.log(uStar / CM)) if (uStar < CM) else (1.0 + 3.0 * np.log(uStar / CM)))
    Fm = np.exp(-0.25 * square(k / KM - 1.0))
    Bh = 0.5 * alpham * CM / c * Fm * Lpm
    a0 = np.log(2.0) / 4.0
    am = 0.13 * uStar / CM
    Delta = np.tanh(a0 + 4.0 * (c / cp)**2.5 + am * (CM / c)**2.5)
    if np.any(Bh < 0):
        print(Bh)
    S = (1.0 / (2.0 * PI)) * k**-4.0 * (Bl + Bh) * (1.0 + Delta)
    dk = 2.0 * PI / u_size
    h = np.sqrt(S / 2.0) * dk

    return k, S, h


x = np.linspace(1, 256, num=256)
k, S, h = phillips(x)
print(np.sum(h), 2 * np.sqrt(20))
plt.plot(x, S, 'k-')
plt.show()
plt.plot(x, h, 'k-')
plt.show()
plt.plot(k, S, 'k-')
plt.show()
plt.plot(k, h, 'k-')
plt.show()

