"""Fungsi model sistem dinamik PM10 Jakarta (dipakai oleh app.py; sama dengan notebook)."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, curve_fit

T_MUSIM = 365.25  # periode gangguan musiman (hari)


def estimasi_k_E0(c):
    """Regresi C(t+1) = a*C(t) + b. Setara Euler dengan dt = 1 hari: k = 1 - a, E0 = b."""
    a, b = np.polyfit(c[:-1], c[1:], 1)
    return 1 - a, b


def gangguan(doy, A, t0):
    """Gangguan musiman d(t) = A * sin(2*pi*(doy - t0) / T)."""
    return A * np.sin(2 * np.pi * (np.asarray(doy, float) - t0) / T_MUSIM)


def kontrol_u(t, jenis="tidak ada", u_maks=0.2, t_mulai=90, t_ramp=90, periode=60):
    """Sinyal kontrol u(t) = fraksi pengurangan emisi (0-1)."""
    t = np.asarray(t, float)
    if jenis == "step":
        return np.where(t >= t_mulai, u_maks, 0.0)
    if jenis == "ramp":
        return u_maks * np.clip((t - t_mulai) / t_ramp, 0.0, 1.0)
    if jenis == "sinusoida":
        return np.where(t >= t_mulai,
                        u_maks / 2 * (1 - np.cos(2 * np.pi * (t - t_mulai) / periode)), 0.0)
    return np.zeros_like(t)


def euler(k, E0, A, t0, C0, u_fn=None, dt=1.0, t_akhir=364, doy_awal=1):
    """Metode Euler: C[n+1] = C[n] + dt * (E0*(1-u) + d - k*C[n])."""
    t = np.arange(0, t_akhir + dt / 2, dt)
    u = np.zeros_like(t) if u_fn is None else u_fn(t)
    d = gangguan(doy_awal + t, A, t0)
    C = np.empty_like(t)
    C[0] = C0
    for n in range(len(t) - 1):
        C[n + 1] = C[n] + dt * (E0 * (1 - u[n]) + d[n] - k * C[n])
    return t, C


def solusi_ivp(k, E0, A, t0, C0, u_fn=None, t_akhir=364, doy_awal=1):
    """Solver SciPy solve_ivp untuk ODE yang sama (dC/dt = E0(1-u) + d - kC)."""
    def rhs(t, C):
        u = 0.0 if u_fn is None else float(u_fn(t))
        return E0 * (1 - u) + float(gangguan(doy_awal + t, A, t0)) - k * C
    t_eval = np.arange(0, t_akhir + 0.5, 1.0)
    sol = solve_ivp(rhs, (0, t_akhir), [C0], t_eval=t_eval, max_step=1.0, rtol=1e-8, atol=1e-8)
    return sol.t, sol.y[0]


def metrik(y, yhat):
    """RMSE, MAE, dan R^2 antara data (y) dan simulasi (yhat)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    sse = np.sum((y - yhat) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return {"RMSE": float(np.sqrt(sse / len(y))), "MAE": float(np.mean(np.abs(y - yhat))), "R2": float(1 - sse / sst)}


def kalibrasi_musiman(y, k, E0, A0, t00):
    """Kalibrasi A dan t0 (k dan E0 dikunci dari regresi) dengan least squares."""
    def sisa(p):
        return euler(k, E0, p[0], p[1], y[0], t_akhir=len(y) - 1)[1] - y
    hasil = least_squares(sisa, [A0, t00], bounds=([0.0, -200.0], [30.0, 400.0]))
    return float(hasil.x[0]), float(hasil.x[1])


def fit_sinus_musiman(doy, y):
    """Fit y = c + amp*sin(2*pi*(doy - ph)/T). Mengembalikan (c, amp, ph)."""
    f = lambda d, c, amp, ph: c + amp * np.sin(2 * np.pi * (d - ph) / T_MUSIM)
    p, _ = curve_fit(f, doy, y, p0=[float(np.mean(y)), 10.0, 110.0])
    return float(p[0]), float(p[1]), float(p[2])
