"""Simulator Dinamika PM10 Jakarta (TA-07 Pemodelan dan Simulasi)."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from model import (estimasi_k_E0, fit_sinus_musiman, kalibrasi_musiman,
                   euler, solusi_ivp, kontrol_u, metrik)

DATA = Path(__file__).parent / "data" / "ispu_dki_all.csv"

st.set_page_config(page_title="Simulator PM10 Jakarta", page_icon="🌫️", layout="wide")


@st.cache_data
def muat_data():
    df = pd.read_csv(DATA, parse_dates=["tanggal"]).set_index("tanggal")
    return df["pm10"]


@st.cache_data
def parameter_awal():
    s = muat_data()
    y = s.loc["2022"].values
    k, E0 = estimasi_k_E0(y)
    x = s.dropna()
    _, amp, ph = fit_sinus_musiman(x.index.dayofyear.values.astype(float), x.values)
    A, t0 = kalibrasi_musiman(y, k, E0, k * amp, ph)
    return {"k": float(k), "E0": float(E0), "A": float(A), "t0": float(t0), "C0": float(y[0])}


s_all = muat_data()
y22 = s_all.loc["2022"]
P = parameter_awal()
tanggal = y22.index

# ---------------- Sidebar ----------------
st.sidebar.header("Parameter model")
k = st.sidebar.slider("Laju pembersihan k (1/hari)", 0.05, 1.50, round(P["k"], 3), 0.005)
E0 = st.sidebar.slider("Emisi dasar E0 (indeks/hari)", 0.0, 100.0, round(P["E0"], 1), 0.1)
A = st.sidebar.slider("Amplitudo gangguan musiman A", 0.0, 15.0, round(P["A"], 2), 0.05)
t0 = st.sidebar.slider("Fase musiman t0 (hari)", 0.0, 365.0, round(P["t0"], 1), 0.5)
C0 = st.sidebar.slider("Kondisi awal C0", 0.0, 120.0, round(P["C0"], 1), 1.0)

st.sidebar.header("Skenario What-If")
jenis = st.sidebar.selectbox("Jenis pengurangan emisi", ["step", "ramp", "sinusoida"])
u_maks = st.sidebar.slider("Besar pengurangan emisi u", 0.0, 0.8, 0.2, 0.05)
t_mulai = st.sidebar.slider("Mulai pada hari ke-", 0, 364, 90, 1)
t_ramp, periode = 90, 60
if jenis == "ramp":
    t_ramp = st.sidebar.slider("Durasi ramp (hari)", 10, 180, 90, 5)
if jenis == "sinusoida":
    periode = st.sidebar.slider("Periode sinusoida (hari)", 10, 180, 60, 5)

st.sidebar.header("Metode numerik")
dt = st.sidebar.selectbox("Step size (dt, hari)", [1.0, 0.5, 0.1], index=0)
bandingkan_ivp = st.sidebar.checkbox("Bandingkan dengan solve_ivp", value=False)

# ---------------- Simulasi ----------------
u_fn = lambda tt: kontrol_u(tt, jenis, u_maks, t_mulai, t_ramp, periode)
langkah = int(round(1 / dt))

_, C_base_full = euler(k, E0, A, t0, C0, None, dt=dt)
_, C_scen_full = euler(k, E0, A, t0, C0, u_fn, dt=dt)
C_base, C_scen = C_base_full[::langkah], C_scen_full[::langkah]

plot = pd.DataFrame({"Data PM10 2022": y22.values,
                     "Baseline (u = 0)": C_base,
                     f"Skenario {jenis}": C_scen}, index=tanggal)
if bandingkan_ivp:
    _, C_ivp = solusi_ivp(k, E0, A, t0, C0, u_fn)
    plot["Skenario (solve_ivp)"] = C_ivp

# ---------------- Tampilan ----------------
st.title("🌫️ Simulator Dinamika PM10 Jakarta")
st.caption("TA-07 Pemodelan dan Simulasi | WIFA SAPUTRA MAHENDRA - 301240063 | "
           "Model ODE orde satu: dC/dt = E0(1-u) + d(t) - kC, diselesaikan dengan metode Euler")

tab1, tab2, tab3 = st.tabs(["Simulasi", "Evaluasi model", "Tentang model"])

with tab1:
    c_star_b = E0 / k
    c_star_s = E0 * (1 - u_maks) / k
    sesudah = slice(min(t_mulai + 30, 364), None)
    rata_b, rata_s = C_base[sesudah].mean(), C_scen[sesudah].mean()
    m1, m2, m3 = st.columns(3)
    m1.metric("Steady state baseline C*", f"{c_star_b:.1f}")
    m2.metric("Steady state skenario C*", f"{c_star_s:.1f}", f"{(c_star_s / c_star_b - 1) * 100:.1f}%")
    m3.metric("Rata-rata PM10 setelah 30 hari skenario", f"{rata_s:.1f}", f"{(rata_s / rata_b - 1) * 100:.1f}% vs baseline")
    st.line_chart(plot)
    if dt != 1.0:
        st.info("Catatan: nilai k hasil regresi dihitung untuk langkah 1 hari. "
                "Mengubah dt dengan k yang sama akan menggeser hasil (lihat bagian Euler vs solve_ivp di notebook).")
    waktu_paruh = np.log(2) / k
    st.write(f"Waktu paruh pembersihan: **{waktu_paruh:.2f} hari**. "
             "Karena responsnya cepat, perubahan emisi langsung terlihat pada PM10 dalam hitungan hari.")

with tab2:
    ma30 = y22.rolling(30, center=True, min_periods=15).mean().values
    bln_data = y22.groupby(y22.index.month).mean().values
    bln_sim = pd.Series(C_base, index=tanggal).groupby(tanggal.month).mean().values
    baris = {
        "vs data harian": metrik(y22.values, C_base),
        "vs rata-rata 30 hari": metrik(ma30, C_base),
        "vs rata-rata bulanan": metrik(bln_data, bln_sim),
    }
    st.subheader("Baseline vs data 2022")
    st.dataframe(pd.DataFrame(baris).T.round(3))
    st.write("Model menangkap **pola musiman**, bukan fluktuasi harian (cuaca harian tidak ada di model). "
             "Karena itu R2 terhadap data harian rendah, sedangkan terhadap rata-rata bulanan tinggi.")
    st.subheader("Data mentah 2022")
    st.dataframe(y22.rename("PM10").to_frame())

with tab3:
    st.markdown("**Stock** = konsentrasi PM10 *C(t)*. **Inflow** = emisi dan gangguan musiman, "
                "**outflow** = pembersihan alami sebanding dengan *C* (umpan balik negatif).")
    st.latex(r"\frac{dC}{dt} = E_0\,(1-u(t)) + d(t) - k\,C")
    st.latex(r"d(t) = A \sin\!\left(\frac{2\pi (t-t_0)}{365{,}25}\right)")
    st.latex(r"C_{n+1} = C_n + \Delta t\,[E_0(1-u_n) + d_n - k\,C_n]")
    st.latex(r"C^* = \frac{E_0 (1-u)}{k}")
    st.markdown("**Sumber data:** ISPU DKI Jakarta (Kaggle, senadu34), tahun 2022.  \n"
                "**Keterbatasan:** nilai ISPU berupa indeks; stasiun bergantian antar hari; satu tahun data; "
                "tanpa variabel cuaca harian. Hasil adalah simulasi perilaku sistem, bukan prediksi kebijakan.")
