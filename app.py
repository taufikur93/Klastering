 import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score

# ========================== 1️⃣ SETUP STREAMLIT ==========================
st.title("🔍 Simulasi RFM & Klasterisasi")

# Upload Dataset
uploaded_file = st.file_uploader("📂 Upload dataset CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📌 Data Awal:")
    st.dataframe(df.head())

    # ========================== 2️⃣ HITUNG RFM ==========================
    st.subheader("📊 Perhitungan RFM")
    
    # Pastikan dataset memiliki kolom yang diperlukan
    if {'CustomerID', 'InvoiceDate', 'TotalAmount'}.issubset(df.columns):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        snapshot_date = df['InvoiceDate'].max()  # Ambil tanggal terbaru
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
            'CustomerID': 'count',  # Frequency
            'TotalAmount': 'sum'  # Monetary
        }).rename(columns={'InvoiceDate': 'Recency', 'CustomerID': 'Frequency', 'TotalAmount': 'Monetary'})
        
        st.dataframe(rfm.head())

        # ========================== 3️⃣ NORMALISASI DATA ==========================
        st.subheader("🔢 Normalisasi Data")
        scaler = MinMaxScaler()
        rfm_normalized = pd.DataFrame(scaler.fit_transform(rfm), columns=['Recency', 'Frequency', 'Monetary'])
        st.write("📌 Data Setelah Normalisasi:")
        st.dataframe(rfm_normalized.head())

        # ========================== 4️⃣ PILIH METODE KLASTERISASI ==========================
        st.subheader("🚀 Metode Klasterisasi")
        method = st.selectbox("🔘 Pilih metode:", ["K-Means", "DBSCAN", "Agglomerative"])
        
        # ========================== 5️⃣ K-MEANS ==========================
        if method == "K-Means":
            k = st.slider("Pilih jumlah klaster (K)", 2, 10, 3)
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(rfm_normalized)
            rfm_normalized["Cluster"] = clusters
            
        # ========================== 6️⃣ DBSCAN ==========================
        elif method == "DBSCAN":
            eps = st.slider("Pilih nilai Epsilon (ε)", 0.1, 2.0, 0.5)
            min_samples = st.slider("Pilih nilai MinPts", 2, 10, 3)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(rfm_normalized)
            rfm_normalized["Cluster"] = clusters

        # ========================== 7️⃣ AGGLOMERATIVE CLUSTERING ==========================
        elif method == "Agglomerative":
            k = st.slider("Pilih jumlah klaster (K)", 2, 10, 3)
            agglo = AgglomerativeClustering(n_clusters=k)
            clusters = agglo.fit_predict(rfm_normalized)
            rfm_normalized["Cluster"] = clusters
        
        # ========================== 8️⃣ VISUALISASI HASIL KLASTER ==========================
        st.subheader("📈 Visualisasi Hasil Klasterisasi")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=rfm_normalized["Recency"], y=rfm_normalized["Monetary"], hue=rfm_normalized["Cluster"], palette="viridis", ax=ax)
        ax.set_xlabel("Recency")
        ax.set_ylabel("Monetary")
        ax.set_title("Hasil Klasterisasi")
        st.pyplot(fig)
        
        # ========================== 9️⃣ EVALUASI DBI ==========================
        st.subheader("📊 Evaluasi Davies-Bouldin Index (DBI)")
        if len(set(clusters)) > 1:  # DBI tidak dapat dihitung jika hanya ada 1 klaster
            dbi_score = davies_bouldin_score(rfm_normalized[['Recency', 'Frequency', 'Monetary']], clusters)
            st.write(f"📌 Nilai DBI untuk metode {method}: **{dbi_score:.3f}**")
        else:
            st.write("⚠️ Tidak dapat menghitung DBI karena hanya ada satu klaster.")

        # ==========================  🔥 SELESAI  ==========================

