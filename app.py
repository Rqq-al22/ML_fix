import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

# =========================
# LOAD DATA
# =========================
st.title("🧠 Mental Wellness Clustering App")

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'wellnes1.csv')

df_original = pd.read_csv(file_path, sep=';')

# =========================
# TAMPILKAN DATA
# =========================
st.subheader("📊 Data Preview (4 Features)")

fitur = ['Stress Level', 'Productivity', 'Sleep Quality', 'Screen Time']
st.dataframe(df_original[fitur])

st.divider()

# =========================
# PREPROCESSING
# =========================
X = df_original[fitur].values

# =========================
# CLUSTERING
# =========================
model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)

df_original['Cluster'] = labels

# =========================
# RATA-RATA PER CLUSTER
# =========================
cluster_summary = df_original.groupby('Cluster').mean(numeric_only=True)

st.subheader("📈 Cluster Summary")
st.dataframe(cluster_summary)

st.divider()

# =========================
# INTERPRETASI CLUSTER
# =========================
st.subheader("🧠 Detailed Interpretation of Agglomerative Clusters (K=3)")

def show_cluster(title, content):
    st.markdown(f"""
    <div style="
        border:2px solid #4CAF50;
        border-radius:10px;
        padding:15px;
        margin-bottom:15px;
        background-color:#f9f9f9;">
        <h4>{title}</h4>
        <pre>{content}</pre>
    </div>
    """, unsafe_allow_html=True)

# Cluster 0
show_cluster("Cluster 0", """
Average Stress Level: 8.93
Average Productivity: 48.89
Average Sleep Quality: 1.21
Average Screen Time: 9.67
Average Mental Wellness Index: 12.76

Higher stress, lower productivity & poor sleep → mental wellness rendah
""")

# Cluster 1
show_cluster("Cluster 1", """
Average Stress Level: 1.64
Average Productivity: 98.06
Average Sleep Quality: 3.29
Average Screen Time: 4.07
Average Mental Wellness Index: 84.71

Low stress, high productivity & good sleep → mental wellness tinggi
""")

# Cluster 2
show_cluster("Cluster 2", """
Average Stress Level: 5.08
Average Productivity: 75.70
Average Sleep Quality: 2.09
Average Screen Time: 6.46
Average Mental Wellness Index: 49.86

Moderate condition → cukup stabil tapi belum optimal
""")
