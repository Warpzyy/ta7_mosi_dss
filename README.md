# TA-07: Simulasi Sistem Dinamik Akumulasi PM10 Jakarta

Nama/NIM: WIFA SAPUTRA MAHENDRA - 301240063 | Mata kuliah: Pemodelan dan Simulasi

Model ODE orde satu dengan umpan balik negatif:

    dC/dt = E0 (1 - u(t)) + d(t) - k C

disimulasikan dengan metode Euler dan dibandingkan dengan data ISPU DKI Jakarta 2022.

## Isi repositori

| File | Fungsi |
|---|---|
| `TA07_Simulasi_PM10_Jakarta.ipynb` | Notebook: model, Euler, `solve_ivp`, kalibrasi, What-If, interpretasi |
| `app.py` | Aplikasi Streamlit (simulator interaktif) |
| `model.py` | Fungsi model yang dipakai `app.py` (sama dengan di notebook) |
| `data/ispu_dki_all.csv` | Data ISPU DKI Jakarta (Kaggle: senadu34) |
| `requirements.txt` | Dependensi Python |

## Menjalankan di komputer sendiri

    pip install -r requirements.txt
    streamlit run app.py

Notebook: buka `TA07_Simulasi_PM10_Jakarta.ipynb` di Jupyter/VS Code/Google Colab, lalu *Run all*. Di Colab, upload dulu `ispu_dki_all.csv` ke folder kerja.

## Deploy (Streamlit Community Cloud)

1. Push seluruh isi folder ini ke repositori GitHub (publik).
2. Buka https://share.streamlit.io lalu masuk dengan akun GitHub.
3. Klik **Create app**, pilih repositori dan branch, isi **Main file path** dengan `app.py`, lalu **Deploy**.
4. Salin link aplikasi (`...streamlit.app`) ke laporan dan README.

Link aplikasi: https://ta7mosidss-warpsyy.streamlit.app/
Link video demo: (isi setelah upload)
