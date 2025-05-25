
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import butter, filtfilt
from io import StringIO

# Butterworth filter
def apply_butterworth_filter(signal, cutoff=10, fs=1000, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

# Offset removal
def remove_offset(signal):
    return signal - np.mean(signal)

# Example: convert to acceleration (dummy scaling for placeholder)
def convert_to_acceleration(raw_signal):
    return raw_signal * 9.81 / 1024

# Dummy parameter computation for demonstration
def compute_params1D(signal, T):
    peak_index = np.argmax(signal)
    F = 1 / T if T > 0 else 0
    R = peak_index * T
    C = np.max(signal) / (np.mean(np.abs(signal)) + 1e-6)
    D = np.log(np.abs(signal[peak_index]) / (np.abs(signal[-1]) + 1e-6))
    return F, R, C, D

# Main Streamlit app
st.title("Analiza semnalului Muston - cu Butterworth și Export Excel")

uploaded_file = st.file_uploader("Incarca fisierul CSV", type=["csv"])

if uploaded_file is not None:
    content = uploaded_file.read().decode('utf-8')
    lines = content.split('\n')

    # Extract sample period T from lines 0-6
    T_line = [line for line in lines[:6] if "Sample Period" in line]
    if T_line:
        T = float(T_line[0].split(",")[-1].strip()) / 1000
    else:
        T = 0.001

    # Read signal data from row 10 onward
    data = pd.read_csv(StringIO(content), skiprows=9)
    channels = ['A0', 'A1', 'A2', 'A3', 'A4']
    time = data["Time_s"]

    df_params = pd.DataFrame(columns=["Canal", "F [Hz]", "R [s]", "C [-]", "D [-]"])

    for ch in channels:
        raw = data[ch]
        acc = convert_to_acceleration(raw)
        acc = remove_offset(acc)
        acc = apply_butterworth_filter(acc, cutoff=10, fs=1/T, order=2)

        F, R, C, D = compute_params1D(acc, T)
        df_params.loc[len(df_params)] = [ch, round(F, 2), round(R, 4), round(C, 3), round(D, 3)]

        # Plot signal
        fig, ax = plt.subplots()
        ax.plot(time, acc)
        ax.set_title(f"Semnal filtrat - {ch}")
        st.pyplot(fig)

    # Show parameter table
    st.subheader("Parametri extrasi")
    st.dataframe(df_params)

    # Save to Excel
    df_params.to_excel("parametri_extrasi.xlsx", index=False)
    st.success("Parametrii au fost salvati in 'parametri_extrasi.xlsx'")
