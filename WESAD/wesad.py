import os
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks

# ========================================================
# 1. CONFIGURACIÓN (WESAD FS = 700 Hz)
# ========================================================
# La "r" soluciona el error unicodeescape en Windows
DATA_PATH = r"C:\Users\Alumnos\Downloads\ReconocimientoPatrones\WESAD"
FS = 700
WINDOW_SEC = 120 
STEP_SEC = 60

SUBJECT_IDS = [f"S{i}" for i in range(2, 12)] + [f"S{i}" for i in range(13, 18)]

# ========================================================
# 2. FUNCIONES DE EXTRACCIÓN (EDA)
# ========================================================
def tonic_phasic(eda, fs):
    eda = eda[~np.isnan(eda)]
    b, a = butter(4, 0.05 / (fs / 2), btype="low")
    tonic = filtfilt(b, a, eda)
    phasic = eda - tonic
    return tonic, phasic

def extract_eda_features(eda, fs):
    tonic, phasic = tonic_phasic(eda, fs)
    peaks, _ = find_peaks(phasic, height=0.01, distance=fs)
    
    # Solución para el error de NumPy 2.0+ (trapz vs trapezoid)
    try:
        auc = np.trapezoid(np.abs(phasic)) / fs
    except AttributeError:
        auc = np.trapz(np.abs(phasic)) / fs
        
    return len(peaks), auc, np.mean(tonic)

# ========================================================
# 3. FUNCIONES DE EXTRACCIÓN (HRV - Corazón) VERSIÓN EXTENDIDA
# ========================================================
def build_hrv_table(subject):
    ecg = subject["signal"]["chest"]["ECG"]
    labels = subject["label"]
    
    cleaned = nk.ecg_clean(ecg, sampling_rate=FS)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=FS)
    rpeaks_idx = rpeaks["ECG_R_Peaks"]

    WINDOW = FS * WINDOW_SEC
    STEP = FS * STEP_SEC
    rows = []

    for start in range(0, len(ecg) - WINDOW, STEP):
        end = start + WINDOW
        peaks = rpeaks_idx[(rpeaks_idx >= start) & (rpeaks_idx < end)] - start
        if len(peaks) <= 2: continue

        rr_ms = np.diff(peaks) / FS * 1000
        try:
            rpeaks_clean = nk.intervals_to_peaks(rr_ms, sampling_rate=FS)
            hrv_t = nk.hrv_time(rpeaks_clean, sampling_rate=FS)
            hrv_f = nk.hrv_frequency(rpeaks_clean, sampling_rate=FS)

            label_bin = np.bincount((labels[start:end] == 2).astype(int)).argmax()

            # ¡AQUÍ ESTÁ LA MAGIA! Agregamos MeanNN, LF y HF
            rows.append({
                "Time": (start + end) / 2 / FS,
                "Label": label_bin,
                "HRV_RMSSD": hrv_t["HRV_RMSSD"].values[0],
                "HRV_SDNN": hrv_t["HRV_SDNN"].values[0],
                "HRV_MeanNN": hrv_t["HRV_MeanNN"].values[0], # Novedad
                "HRV_LF": hrv_f["HRV_LF"].values[0],         # Novedad
                "HRV_HF": hrv_f["HRV_HF"].values[0],         # Novedad
                "HRV_LFHF": hrv_f["HRV_LFHF"].values[0]
            })
        except: continue
    return pd.DataFrame(rows)

# ========================================================
# 4. PROCESAMIENTO GLOBAL
# ========================================================
eda_all, hrv_all = [], []

for sid in SUBJECT_IDS:
    file_path = os.path.join(DATA_PATH, sid, f"{sid}.pkl")
    if not os.path.exists(file_path):
        print(f"Archivo no encontrado: {file_path}")
        continue
        
    print(f"Procesando Sujeto: {sid}...")
    with open(file_path, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    # --- Procesar EDA ---
    eda = subject["signal"]["chest"]["EDA"].flatten()
    labels = subject["label"].flatten()
    rows_eda = []
    
    WINDOW = FS * WINDOW_SEC
    STEP = FS * STEP_SEC
    
    for i in range(0, len(eda) - WINDOW, STEP):
        # Etiquetado binario: 1 si es Stress (etiqueta 2), 0 en otro caso
        label_bin = 1 if np.bincount(labels[i:i+WINDOW]).argmax() == 2 else 0
        scr, auc, tonic = extract_eda_features(eda[i:i+WINDOW], FS)
        rows_eda.append({
            "Time": i / FS, "Label": label_bin, "Subject": sid,
            "EDA_SCR": scr, "EDA_AUC": auc, "EDA_Tonic": tonic
        })
    
    eda_all.append(pd.DataFrame(rows_eda))
    hrv_all.append(build_hrv_table(subject).assign(Subject=sid))

# ========================================================
# 5. FUSIÓN Y GUARDADO
# ========================================================
if eda_all and hrv_all:
    final_df = pd.merge(
        pd.concat(eda_all), 
        pd.concat(hrv_all), 
        on=["Time", "Label", "Subject"], 
        how="inner"
    )
    final_df.to_csv("wesad_features_punto1.csv", index=False)
    print("\n¡Proceso completado! Archivo generado: wesad_features_punto1.csv")
else:
    print("\nNo se pudieron procesar los datos. Revisa las rutas.")