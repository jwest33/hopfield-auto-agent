#!/usr/bin/env python
"""
streamlit_hopfield_weather.py
------------------------------
Interactive Streamlit app for exploring delay-embedded temperature
changes and strange attractors using Modern Hopfield memory.
Enhanced with interpretive descriptions for each section.

Install: pip install streamlit meteostat numpy pandas scikit-learn scipy matplotlib plotly
Run:     streamlit run streamlit_hopfield_weather.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from meteostat import Point, Daily
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
import plotly.express as px
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
CITIES = {
    "Phoenix, AZ (PHX)": Point(33.4484, -112.0740),
    "Denver, CO (DEN)": Point(39.7392, -104.9903),
    "Seattle, WA (SEA)": Point(47.6062, -122.3321),
}
CITY_CODES = {v: k.split("(")[-1][:-1] for k, v in CITIES.items()}
YEARS_BACK = 5
EMBED_DIM = 8
TAU = 1
PCA_DIM = 3
BETA = 2.0
NOISE_STD = 0.3
MAX_SAMPLE = 4000

# ---------------- UTILITIES -------------

def fetch_city_deltas(point, years):
    end = datetime.utcnow() - timedelta(days=1)
    start = end - timedelta(days=365 * years)
    df = Daily(point, start, end).fetch()
    ts = df["tavg"].dropna()
    return ts.diff().dropna()


def time_delay_embed(series, dim, tau):
    arr = series.values
    N = len(arr) - (dim - 1) * tau
    return np.column_stack([arr[i:i + N] for i in range(0, dim * tau, tau)])

# ---------------- HOPFIELD --------------

class ModernHopfield:
    def __init__(self, beta=1.0, iters=3):
        self.beta = beta
        self.iters = iters
        self.M = None

    def store(self, patterns):
        self.M = patterns.astype(np.float32)

    def recall(self, x):
        v = x.astype(np.float32).copy()
        for _ in range(self.iters):
            v = softmax(self.beta * (self.M @ v)) @ self.M
        return v

    def energy(self, x):
        return -logsumexp(self.beta * (self.M @ x.astype(np.float32)))


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ---------------- FRACTAL DIMENSION -----

def correlation_dimension(data, max_num_r=20):
    dists = cdist(data, data)
    np.fill_diagonal(dists, np.inf)
    flat = dists[np.isfinite(dists)]
    r_min, r_max = np.percentile(flat, [5, 95])
    r_min = max(r_min, flat[flat > 0].min())
    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), max_num_r)
    N = len(data)
    log_r, log_C = [], []
    for r in r_vals:
        C = (dists < r).sum() / (N * (N - 1))
        if C > 0:
            log_r.append(np.log(r)); log_C.append(np.log(C))
    if len(log_r) < 2:
        return np.nan, ([], [])
    return np.polyfit(log_r, log_C, 1)[0], (log_r, log_C)

# ---------------- STREAMLIT -------------
st.set_page_config(page_title="Hopfield Attractors – Weather ΔT", layout="wide")
st.title("🧠 Hopfield Memory on City Temperature Patterns")

st.markdown("""
This tool demonstrates how **Hopfield memory networks** can store and recall
patterns in day-to-day **temperature changes (ΔT)** across different cities.
We use a technique called **delay embedding** to reconstruct the underlying
dynamics and then analyze their **fractal structure**.
""")

city_name = st.selectbox("Choose a city to analyze", list(CITIES.keys()))
point = CITIES[city_name]
city_code = CITY_CODES[point]

# Load and process data
with st.spinner("Loading weather data and building embeddings..."):
    series = fetch_city_deltas(point, YEARS_BACK)
    embedded = time_delay_embed(series, EMBED_DIM, TAU)
    scaler = StandardScaler(); embedded = scaler.fit_transform(embedded)
    pca = PCA(n_components=PCA_DIM); reduced = pca.fit_transform(embedded)

    hop = ModernHopfield(BETA, iters=5)
    hop.store(embedded)

    index = st.slider("Select vector index", 0, len(embedded) - 1, 1000)
    clean = embedded[index]
    noisy = clean + np.random.normal(scale=NOISE_STD, size=clean.shape)
    recalled = hop.recall(noisy)
    energy = hop.energy(recalled)

    sample = embedded[np.random.choice(len(embedded), min(len(embedded), MAX_SAMPLE), False)]
    dim, (log_r, log_C) = correlation_dimension(sample)

# Time series panel
st.subheader("📉 Reconstructed Delay-Embedded Temperature Change Vector")
st.markdown("""
This shows a **single embedded pattern** from your selected city:
- **Clean**: The original delay-embedded ΔT vector.
- **Noisy**: After adding random Gaussian noise.
- **Recalled**: What the Hopfield network reconstructs.

A close match between clean and recalled suggests the network recognizes this
pattern and has successfully recalled it.
""")
st.line_chart(pd.DataFrame({"clean": clean, "noisy": noisy, "recalled": recalled}))

# Fractal dimension panel
st.subheader("📏 Fractal (Correlation) Dimension of the Attractor")
st.markdown("""
This plot estimates the **fractal dimension** of the attractor formed by all
embedded ΔT vectors. A higher dimension implies more complex and
chaotic dynamics, whereas a value near 1 suggests near-periodic behavior.
""")
if log_r:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=log_r,
        y=log_C,
        mode='markers+lines',
        marker=dict(color='tomato'),
        line=dict(dash='dot'),
        name='log C(r)'
    ))
    fig.update_layout(
        title=f"Estimated correlation dimension ≈ {dim:.2f}",
        xaxis_title="log r",
        yaxis_title="log C(r)",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Not enough structure to estimate dimension.")

st.subheader("🌀 2D Projections of the PCA Attractor")
st.markdown("""
3D attractor plots can be unstable on some browsers. These 2D projections show
different views of the underlying geometry in the reduced PCA space.
""")

fig_p1 = px.line(x=reduced[:, 0], y=reduced[:, 1],
                 labels={'x': 'PC1', 'y': 'PC2'}, title='PC1 vs PC2')
fig_p2 = px.line(x=reduced[:, 0], y=reduced[:, 2],
                 labels={'x': 'PC1', 'y': 'PC3'}, title='PC1 vs PC3')
fig_p3 = px.line(x=reduced[:, 1], y=reduced[:, 2],
                 labels={'x': 'PC2', 'y': 'PC3'}, title='PC2 vs PC3')

st.plotly_chart(fig_p1, use_container_width=True)
st.plotly_chart(fig_p2, use_container_width=True)
st.plotly_chart(fig_p3, use_container_width=True)
