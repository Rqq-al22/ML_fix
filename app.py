import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

st.set_page_config(layout="wide")
st.title("🧠 Mental Wellness Clustering App (Agglomerative K=3)")

# =========================
# LOAD DATA (AMAN UNTUK DEPLOY)
# =========================
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'wellnes1.csv')

df_original = pd.read_csv(file_path, sep=';')

# rapihin nama kolom biar gak error
df_original.columns = df_original.columns.str.strip()

# =========================
# FITUR
# =========================
selected_features = [
    'stress_level_0_10',
    'productivity_0_100',
    'sleep_quality_1_5',
    'screen_time_hours'
]

all_features_for_comparison = selected_features + ['mental_wellness_index_0_100']

# =========================
# TAMPIL DATA
# =========================
st.subheader("📊 Data Preview (4 Features)")
st.dataframe(df_original[selected_features])

st.divider()

# =========================
# SCALING + CLUSTERING
# =========================
scaler = StandardScaler()
df_selected = df_original[selected_features]

scaler.fit(df_selected)
df_scaled = pd.DataFrame(scaler.transform(df_selected), columns=selected_features)

model = AgglomerativeClustering(n_clusters=3, linkage='average')
labels = model.fit_predict(df_scaled)

df_original['cluster'] = labels

# =========================
# CENTROID MANUAL
# =========================
centroids = []
for i in range(3):
    cluster_points = df_scaled[labels == i]
    centroids.append(cluster_points.mean(axis=0))
centroids = np.array(centroids)

# =========================
# SUMMARY
# =========================
cluster_summary = df_original.groupby('cluster')[all_features_for_comparison].mean().round(2)
overall_means = df_original[all_features_for_comparison].mean().round(2)

st.subheader("📈 Cluster Summary")
st.dataframe(cluster_summary)

st.divider()

# =========================
# FUNGSI PREDIKSI
# =========================
def predict(data_scaled):
    distances = cdist([data_scaled], centroids)
    return distances.argmin(), distances.min()

# =========================
# INPUT USER
# =========================
st.subheader("🎯 Input Data Anda")

with st.form("form_input"):
    col1, col2 = st.columns(2)

    with col1:
        stress = st.slider("Stress Level (0-10)", 0.0, 10.0, float(overall_means['stress_level_0_10']))
        productivity = st.slider("Productivity (0-100)", 0.0, 100.0, float(overall_means['productivity_0_100']))

    with col2:
        sleep = st.slider("Sleep Quality (1-5)", 1.0, 5.0, float(overall_means['sleep_quality_1_5']))
        screen = st.slider("Screen Time (hours)", 0.0, 24.0, float(overall_means['screen_time_hours']))

    submit = st.form_submit_button("Prediksi")

# =========================
# HASIL
# =========================
if submit:
    input_data = np.array([[stress, productivity, sleep, screen]])
    input_scaled = scaler.transform(input_data)

    cluster, dist = predict(input_scaled[0])

    st.success(f"✅ Masuk ke **Cluster {cluster}**")
    st.info(f"Jarak ke centroid: {dist:.2f}")

    st.divider()

    # =========================
    # BOX INTERPRETASI
    # =========================
    def box(title, text):
        st.markdown(f"""
        <div style="
            border:2px solid #4CAF50;
            border-radius:10px;
            padding:15px;
            margin-bottom:15px;
            background-color:#f9f9f9;">
            <h4>{title}</h4>
            <pre>{text}</pre>
        </div>
        """, unsafe_allow_html=True)

    if cluster == 0:
        box("Cluster 0 (Stress Tinggi)", """
Stress tinggi, produktivitas rendah, tidur buruk
→ Mental wellness rendah
""")

    elif cluster == 1:
        box("Cluster 1 (Sehat & Produktif)", """
Stress rendah, produktivitas tinggi, tidur bagus
→ Mental wellness tinggi
""")

    elif cluster == 2:
        box("Cluster 2 (Moderate)", """
Kondisi sedang
→ cukup stabil tapi belum optimal
""")

    st.subheader("📊 Detail Cluster")
    info = cluster_summary.loc[cluster]

    for col in all_features_for_comparison:
        st.write(f"{col} : {info[col]:.2f}")
