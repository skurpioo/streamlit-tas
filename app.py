# ===========================================================
# APLIKASI MACHINE LEARNING DENGAN STREAMLIT
# TAS Mata Kuliah AI - UKSW
# Regresi, Klasifikasi, Clustering + MongoDB
# ===========================================================

# --- Import library ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import io

# Scikit-learn: datasets
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Scikit-learn: model regresi
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Scikit-learn: model klasifikasi
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Scikit-learn: model clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Scikit-learn: metrics regresi
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Scikit-learn: metrics klasifikasi
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Scikit-learn: metrics clustering
from sklearn.metrics import silhouette_score

# MongoDB
from pymongo import MongoClient

warnings.filterwarnings("ignore")

# --- Konfigurasi halaman ---
st.set_page_config(
    page_title="ML App - TAS AI",
    page_icon="🤖",
    layout="wide",
)

# ===========================================================
# KONEKSI MONGODB
# ===========================================================

@st.cache_resource
def get_mongo_client():
    """Buat koneksi MongoDB."""
    try:
        uri = st.secrets["mongo"]["uri"]
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Test koneksi
        client.admin.command("ping")
        return client
    except Exception as e:
        st.error(f"❌ Gagal koneksi MongoDB: {e}")
        return None


def get_database():
    """Ambil database dari MongoDB."""
    client = get_mongo_client()
    if client:
        db_name = st.secrets["mongo"]["db_name"]
        return client[db_name]
    return None


# ===========================================================
# FUNGSI-FUNGSI HELPER
# ===========================================================

@st.cache_data
def load_regression_data(nama_dataset):
    """Muat dataset regresi dari scikit-learn."""
    if nama_dataset == "Diabetes":
        data = datasets.load_diabetes()
    elif nama_dataset == "California Housing":
        data = datasets.fetch_california_housing()
    elif nama_dataset == "Linnerud":
        data = datasets.load_linnerud()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target[:, 0]
        return df, data.feature_names, "Chins"

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data.feature_names, "target"


@st.cache_data
def load_classification_data(nama_dataset):
    """Muat dataset klasifikasi dari scikit-learn."""
    if nama_dataset == "Iris":
        data = datasets.load_iris()
    elif nama_dataset == "Wine":
        data = datasets.load_wine()
    elif nama_dataset == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif nama_dataset == "Digits":
        data = datasets.load_digits()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    target_names = data.target_names
    return df, data.feature_names, target_names


@st.cache_data
def load_clustering_data(nama_dataset):
    """Muat dataset untuk clustering dari scikit-learn."""
    if nama_dataset == "Iris":
        data = datasets.load_iris()
    elif nama_dataset == "Wine":
        data = datasets.load_wine()
    elif nama_dataset == "Breast Cancer":
        data = datasets.load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df, data.feature_names


def get_regression_model(nama_model):
    """Ambil model regresi berdasarkan nama."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=100),
    }
    return models[nama_model]


def get_classification_model(nama_model):
    """Ambil model klasifikasi berdasarkan nama."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(random_state=42, n_estimators=100),
        "Support Vector Machine": SVC(random_state=42),
    }
    return models[nama_model]


def get_clustering_model(nama_model, n_clusters, eps, min_samples):
    """Ambil model clustering berdasarkan nama."""
    if nama_model == "K-Means":
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif nama_model == "DBSCAN":
        return DBSCAN(eps=eps, min_samples=min_samples)
    elif nama_model == "Agglomerative Clustering":
        return AgglomerativeClustering(n_clusters=n_clusters)


# ===========================================================
# FUNGSI MONGODB CRUD
# ===========================================================

def mongo_create_dataset(nama, deskripsi, tipe, df):
    """Simpan dataset ke MongoDB."""
    db = get_database()
    if db is None:
        return False, "Koneksi MongoDB gagal"

    collection = db["datasets"]

    # Cek apakah nama sudah ada
    if collection.find_one({"nama": nama}):
        return False, f"Dataset '{nama}' sudah ada!"

    # Konversi DataFrame ke records
    records = df.to_dict(orient="records")
    columns = list(df.columns)

    doc = {
        "nama": nama,
        "deskripsi": deskripsi,
        "tipe": tipe,
        "columns": columns,
        "data": records,
        "jumlah_baris": len(df),
        "jumlah_kolom": len(columns),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    collection.insert_one(doc)
    return True, f"Dataset '{nama}' berhasil disimpan! ({len(df)} baris, {len(columns)} kolom)"


def mongo_read_datasets():
    """Ambil semua dataset dari MongoDB."""
    db = get_database()
    if db is None:
        return []

    collection = db["datasets"]
    datasets_list = list(collection.find({}, {"data": 0}))  # Tanpa data untuk list
    return datasets_list


def mongo_read_dataset_full(nama):
    """Ambil satu dataset lengkap dari MongoDB."""
    db = get_database()
    if db is None:
        return None

    collection = db["datasets"]
    doc = collection.find_one({"nama": nama})
    return doc


def mongo_update_dataset(nama_lama, nama_baru, deskripsi_baru):
    """Update info dataset di MongoDB."""
    db = get_database()
    if db is None:
        return False, "Koneksi MongoDB gagal"

    collection = db["datasets"]

    # Cek nama baru sudah ada (jika berbeda)
    if nama_baru != nama_lama and collection.find_one({"nama": nama_baru}):
        return False, f"Dataset '{nama_baru}' sudah ada!"

    result = collection.update_one(
        {"nama": nama_lama},
        {"$set": {"nama": nama_baru, "deskripsi": deskripsi_baru}},
    )
    if result.modified_count > 0:
        return True, f"Dataset berhasil diupdate!"
    return False, "Tidak ada perubahan."


def mongo_delete_dataset(nama):
    """Hapus dataset dari MongoDB."""
    db = get_database()
    if db is None:
        return False, "Koneksi MongoDB gagal"

    collection = db["datasets"]
    result = collection.delete_one({"nama": nama})
    if result.deleted_count > 0:
        return True, f"Dataset '{nama}' berhasil dihapus!"
    return False, "Dataset tidak ditemukan."


# ===========================================================
# FUNGSI KOMENTAR PUBLIK
# ===========================================================

def mongo_add_comment(nama, komentar):
    """Tambah komentar ke MongoDB."""
    db = get_database()
    if db is None:
        return False, "Koneksi MongoDB gagal"

    collection = db["comments"]
    doc = {
        "nama": nama,
        "komentar": komentar,
        "waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    collection.insert_one(doc)
    return True, "Komentar berhasil ditambahkan!"


def mongo_get_comments():
    """Ambil semua komentar dari MongoDB (terbaru dulu)."""
    db = get_database()
    if db is None:
        return []

    collection = db["comments"]
    comments = list(collection.find().sort("waktu", -1))
    return comments


def mongo_delete_comment(comment_id):
    """Hapus komentar dari MongoDB."""
    from bson import ObjectId
    db = get_database()
    if db is None:
        return False

    collection = db["comments"]
    result = collection.delete_one({"_id": ObjectId(comment_id)})
    return result.deleted_count > 0


# ===========================================================
# SIDEBAR
# ===========================================================

st.sidebar.title("⚙️ Pengaturan Model")
st.sidebar.write("Atur parameter di bawah, lalu klik **Train Model** untuk memulai training.")

# Pilih jenis task
jenis_task = st.sidebar.radio(
    "🎯 Pilih Jenis Task",
    ["Regresi", "Klasifikasi", "Clustering", "Manajemen Dataset", "Komentar Publik"],
)

# --- Pengaturan untuk REGRESI ---
if jenis_task == "Regresi":
    # Sumber data
    sumber_data_reg = st.sidebar.radio(
        "📂 Sumber Data",
        ["Scikit-learn", "MongoDB"],
        key="sumber_reg",
    )

    if sumber_data_reg == "Scikit-learn":
        dataset_regresi = st.sidebar.selectbox(
            "📊 Pilih Dataset",
            ["Diabetes", "California Housing", "Linnerud"],
        )
    else:
        # Ambil dataset dari MongoDB yang bertipe regresi
        mongo_datasets = mongo_read_datasets()
        mongo_reg = [d["nama"] for d in mongo_datasets if d.get("tipe") == "Regresi"]
        if mongo_reg:
            dataset_regresi_mongo = st.sidebar.selectbox(
                "📊 Pilih Dataset MongoDB",
                mongo_reg,
            )
        else:
            st.sidebar.warning("Belum ada dataset regresi di MongoDB.")
            dataset_regresi_mongo = None

    model_regresi = st.sidebar.selectbox(
        "🧠 Pilih Model",
        [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
        ],
    )
    test_size = st.sidebar.slider(
        "📐 Test Size", 0.10, 0.50, 0.20, 0.05,
        help="Proporsi data yang digunakan untuk testing",
    )
    random_state = st.sidebar.number_input(
        "🎲 Random State", min_value=0, max_value=1000, value=42, step=1,
    )
    tombol_train = st.sidebar.button("🚀 Train Model", use_container_width=True, type="primary")

# --- Pengaturan untuk KLASIFIKASI ---
elif jenis_task == "Klasifikasi":
    sumber_data_cls = st.sidebar.radio(
        "📂 Sumber Data",
        ["Scikit-learn", "MongoDB"],
        key="sumber_cls",
    )

    if sumber_data_cls == "Scikit-learn":
        dataset_klasifikasi = st.sidebar.selectbox(
            "📊 Pilih Dataset",
            ["Iris", "Wine", "Breast Cancer", "Digits"],
        )
    else:
        mongo_datasets = mongo_read_datasets()
        mongo_cls = [d["nama"] for d in mongo_datasets if d.get("tipe") == "Klasifikasi"]
        if mongo_cls:
            dataset_klasifikasi_mongo = st.sidebar.selectbox(
                "📊 Pilih Dataset MongoDB",
                mongo_cls,
            )
        else:
            st.sidebar.warning("Belum ada dataset klasifikasi di MongoDB.")
            dataset_klasifikasi_mongo = None

    model_klasifikasi = st.sidebar.selectbox(
        "🧠 Pilih Model",
        [
            "Logistic Regression",
            "K-Nearest Neighbors",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "Support Vector Machine",
        ],
    )
    test_size = st.sidebar.slider(
        "📐 Test Size", 0.10, 0.50, 0.20, 0.05,
        help="Proporsi data yang digunakan untuk testing",
    )
    random_state = st.sidebar.number_input(
        "🎲 Random State", min_value=0, max_value=1000, value=42, step=1,
    )
    tombol_train = st.sidebar.button("🚀 Train Model", use_container_width=True, type="primary")

# --- Pengaturan untuk CLUSTERING ---
elif jenis_task == "Clustering":
    sumber_data_clust = st.sidebar.radio(
        "📂 Sumber Data",
        ["Scikit-learn", "MongoDB"],
        key="sumber_clust",
    )

    if sumber_data_clust == "Scikit-learn":
        dataset_clustering = st.sidebar.selectbox(
            "📊 Pilih Dataset",
            ["Iris", "Wine", "Breast Cancer"],
        )
    else:
        mongo_datasets = mongo_read_datasets()
        mongo_clust = [d["nama"] for d in mongo_datasets if d.get("tipe") == "Clustering"]
        if mongo_clust:
            dataset_clustering_mongo = st.sidebar.selectbox(
                "📊 Pilih Dataset MongoDB",
                mongo_clust,
            )
        else:
            st.sidebar.warning("Belum ada dataset clustering di MongoDB.")
            dataset_clustering_mongo = None

    model_clustering = st.sidebar.selectbox(
        "🧠 Pilih Algoritma",
        ["K-Means", "DBSCAN", "Agglomerative Clustering"],
    )

    # Parameter khusus per model
    n_clusters = 3
    eps_val = 0.5
    min_samples_val = 5

    if model_clustering in ["K-Means", "Agglomerative Clustering"]:
        n_clusters = st.sidebar.slider(
            "🔢 Jumlah Cluster (K)", 2, 10, 3, 1,
            help="Jumlah kelompok yang ingin dibentuk",
        )
    if model_clustering == "DBSCAN":
        eps_val = st.sidebar.slider(
            "📏 Epsilon (eps)", 0.1, 5.0, 0.5, 0.1,
            help="Jarak maksimum antar titik dalam satu cluster",
        )
        min_samples_val = st.sidebar.slider(
            "👥 Min Samples", 2, 20, 5, 1,
            help="Jumlah minimum titik untuk membentuk cluster",
        )

    tombol_train = st.sidebar.button("🚀 Train Model", use_container_width=True, type="primary")

# --- Pengaturan untuk MANAJEMEN DATASET ---
elif jenis_task == "Manajemen Dataset":
    operasi_crud = st.sidebar.radio(
        "🔧 Pilih Operasi",
        ["📤 Upload Dataset", "📋 Lihat Semua Dataset", "✏️ Update Dataset", "🗑️ Hapus Dataset"],
    )

# Tips di sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tips:** Coba berbagai kombinasi dataset & model!")

# ===========================================================
# KONTEN UTAMA
# ===========================================================

# ===================== REGRESI =====================
if jenis_task == "Regresi":
    st.title("📈 Regresi Machine Learning")
    st.write("Aplikasi demo Machine Learning menggunakan dataset dari scikit-learn atau MongoDB.")

    st.info("👈 Silakan atur parameter di sidebar, lalu klik **🚀 Train Model** untuk memulai training.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Dataset")
        st.write("Scikit-learn atau MongoDB")
    with col2:
        st.markdown("### 🧠 Model")
        st.write("5 model regresi tersedia")
    with col3:
        st.markdown("### 📐 Evaluasi")
        st.write("Metrics lengkap + visualisasi")

    if tombol_train:
        # Muat data berdasarkan sumber
        if sumber_data_reg == "Scikit-learn":
            df, feature_names, target_name = load_regression_data(dataset_regresi)
            X = df[feature_names]
            y = df["target"] if "target" in df.columns else df[target_name]
            nama_dataset_tampil = dataset_regresi
        else:
            if dataset_regresi_mongo is None:
                st.error("❌ Belum ada dataset regresi di MongoDB!")
                st.stop()
            doc = mongo_read_dataset_full(dataset_regresi_mongo)
            df = pd.DataFrame(doc["data"])
            # Kolom terakhir sebagai target
            feature_names = doc["columns"][:-1]
            target_col = doc["columns"][-1]
            X = df[feature_names]
            y = df[target_col]
            nama_dataset_tampil = f"{dataset_regresi_mongo} (MongoDB)"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
        )

        model = get_regression_model(model_regresi)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.success(f"✅ Model berhasil ditraining dalam {training_time:.3f} detik!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Task", "Regresi")
        col2.metric("📊 Dataset", nama_dataset_tampil)
        col3.metric("🧠 Model", model_regresi.split()[0])
        col4.metric("⏱️ Training Time", f"{training_time:.3f}s")

        tab1, tab2 = st.tabs(["📊 Training & Visualisasi", "📐 Evaluation Metrics"])

        with tab1:
            st.header("📊 Hasil Visualisasi")

            c1, c2 = st.columns(2)
            c1.success(f"Data Training: {len(X_train)} sampel")
            c2.error(f"Data Testing: {len(X_test)} sampel")

            fig_col1, fig_col2 = st.columns(2)

            with fig_col1:
                st.subheader("📈 Actual vs Predicted (Scatter)")
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                ax1.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax1.plot(
                    [min_val, max_val], [min_val, max_val],
                    "r--", linewidth=2, label="Perfect Prediction",
                )
                ax1.set_xlabel("Nilai Aktual")
                ax1.set_ylabel("Nilai Prediksi")
                ax1.set_title("Actual vs Predicted")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)

            with fig_col2:
                st.subheader("📉 Perbandingan per Index (50 pertama)")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                n_show = min(50, len(y_test))
                idx = range(n_show)
                y_test_arr = y_test.values if hasattr(y_test, "values") else np.array(y_test)
                ax2.plot(idx, y_test_arr[:n_show], "o-", label="Actual", markersize=4)
                ax2.plot(idx, y_pred[:n_show], "x-", label="Predicted", markersize=4)
                ax2.set_xlabel("Index")
                ax2.set_ylabel("Nilai")
                ax2.set_title("Actual vs Predicted per Index")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)

            st.subheader("📊 Distribusi Residual")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            y_test_arr = y_test.values if hasattr(y_test, "values") else np.array(y_test)
            residuals = y_test_arr - y_pred
            ax3.hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
            ax3.axvline(x=0, color="red", linestyle="--", linewidth=2)
            ax3.set_xlabel("Residual (Actual - Predicted)")
            ax3.set_ylabel("Frekuensi")
            ax3.set_title("Distribusi Residual")
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)

        with tab2:
            st.header("📐 Evaluation Metrics")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", f"{mae:.4f}", help="Mean Absolute Error")
            m2.metric("MSE", f"{mse:.4f}", help="Mean Squared Error")
            m3.metric("RMSE", f"{rmse:.4f}", help="Root Mean Squared Error")
            m4.metric("R² Score", f"{r2:.4f}", help="Koefisien determinasi (1.0 = sempurna)")

            st.markdown("---")
            st.subheader("📝 Penjelasan Metrics")
            st.markdown(f"""
            | Metric | Nilai | Keterangan |
            |--------|-------|------------|
            | **MAE** | {mae:.4f} | Rata-rata kesalahan absolut. Semakin kecil semakin baik. |
            | **MSE** | {mse:.4f} | Rata-rata kesalahan kuadrat. Sensitif terhadap outlier. |
            | **RMSE** | {rmse:.4f} | Akar dari MSE. Satuan sama dengan target. |
            | **R²** | {r2:.4f} | Proporsi variansi yang dijelaskan model (0-1). Semakin tinggi semakin baik. |
            """)

            st.subheader("🔍 Interpretasi")
            if r2 >= 0.9:
                st.success(f"🌟 Model **sangat baik**! R² = {r2:.4f} (≥ 0.9)")
            elif r2 >= 0.7:
                st.info(f"👍 Model **cukup baik**. R² = {r2:.4f} (≥ 0.7)")
            elif r2 >= 0.5:
                st.warning(f"⚠️ Model **kurang baik**. R² = {r2:.4f} (≥ 0.5). Coba model lain.")
            else:
                st.error(f"❌ Model **buruk**. R² = {r2:.4f} (< 0.5). Ganti model atau dataset.")

            st.subheader("📊 Perbandingan Metrics")
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            metrics_names = ["MAE", "MSE", "RMSE"]
            metrics_values = [mae, mse, rmse]
            colors = ["#2ecc71", "#e74c3c", "#3498db"]
            bars = ax4.bar(metrics_names, metrics_values, color=colors, edgecolor="black")
            for bar, val in zip(bars, metrics_values):
                ax4.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold",
                )
            ax4.set_ylabel("Nilai")
            ax4.set_title("Perbandingan Error Metrics")
            ax4.grid(True, alpha=0.3, axis="y")
            st.pyplot(fig4)

# ===================== KLASIFIKASI =====================
elif jenis_task == "Klasifikasi":
    st.title("🏷️ Klasifikasi Machine Learning")
    st.write("Aplikasi demo Klasifikasi menggunakan dataset dari scikit-learn atau MongoDB.")

    st.info("👈 Silakan atur parameter di sidebar, lalu klik **🚀 Train Model** untuk memulai training.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Dataset")
        st.write("Scikit-learn atau MongoDB")
    with col2:
        st.markdown("### 🧠 Model")
        st.write("5 model klasifikasi tersedia")
    with col3:
        st.markdown("### 📐 Evaluasi")
        st.write("Confusion Matrix + classification report")

    if tombol_train:
        # Muat data berdasarkan sumber
        if sumber_data_cls == "Scikit-learn":
            df, feature_names, target_names = load_classification_data(dataset_klasifikasi)
            X = df[feature_names]
            y = df["target"]
            nama_dataset_tampil = dataset_klasifikasi
        else:
            if dataset_klasifikasi_mongo is None:
                st.error("❌ Belum ada dataset klasifikasi di MongoDB!")
                st.stop()
            doc = mongo_read_dataset_full(dataset_klasifikasi_mongo)
            df = pd.DataFrame(doc["data"])
            feature_names = doc["columns"][:-1]
            target_col = doc["columns"][-1]
            X = df[feature_names]
            y = df[target_col]
            # Encode target jika string
            if y.dtype == object:
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y))
                target_names = le.classes_
            else:
                target_names = [str(v) for v in sorted(y.unique())]
            nama_dataset_tampil = f"{dataset_klasifikasi_mongo} (MongoDB)"

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
        )

        # Scaling untuk model yang butuh (Logistic, KNN, SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = get_classification_model(model_klasifikasi)
        start_time = time.time()

        # Pakai scaled data untuk model tertentu
        if model_klasifikasi in ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        training_time = time.time() - start_time

        # Hitung metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=[str(t) for t in target_names], zero_division=0)

        # Success message
        st.success(f"✅ Model berhasil ditraining dalam {training_time:.3f} detik!")

        # Info ringkasan
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Task", "Klasifikasi")
        col2.metric("📊 Dataset", nama_dataset_tampil)
        col3.metric("🧠 Model", model_klasifikasi.split()[0])
        col4.metric("⏱️ Training Time", f"{training_time:.3f}s")

        # --- Tabs ---
        tab1, tab2 = st.tabs(["📊 Training & Visualisasi", "📐 Evaluation Metrics"])

        # ===== TAB 1: Visualisasi =====
        with tab1:
            st.header("📊 Hasil Visualisasi")

            # Info jumlah data
            c1, c2 = st.columns(2)
            c1.success(f"Data Training: {len(X_train)} sampel")
            c2.error(f"Data Testing: {len(X_test)} sampel")

            # Confusion Matrix
            st.subheader("🔲 Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names,
                ax=ax_cm,
                linewidths=0.5,
                linecolor="gray",
            )
            ax_cm.set_xlabel("Prediksi", fontsize=12)
            ax_cm.set_ylabel("Aktual", fontsize=12)
            ax_cm.set_title("Confusion Matrix", fontsize=14)
            st.pyplot(fig_cm)

            # Distribusi prediksi vs aktual
            st.subheader("📊 Distribusi Kelas: Aktual vs Prediksi")
            fig_dist, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(12, 5))

            # Aktual
            unique_actual, counts_actual = np.unique(y_test, return_counts=True)
            ax_d1.bar(
                [str(target_names[i]) for i in unique_actual],
                counts_actual,
                color="#3498db",
                edgecolor="black",
            )
            ax_d1.set_title("Distribusi Kelas Aktual")
            ax_d1.set_ylabel("Jumlah")
            ax_d1.grid(True, alpha=0.3, axis="y")

            # Prediksi
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            ax_d2.bar(
                [str(target_names[i]) for i in unique_pred],
                counts_pred,
                color="#2ecc71",
                edgecolor="black",
            )
            ax_d2.set_title("Distribusi Kelas Prediksi")
            ax_d2.set_ylabel("Jumlah")
            ax_d2.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            st.pyplot(fig_dist)

        # ===== TAB 2: Evaluation Metrics =====
        with tab2:
            st.header("📐 Evaluation Metrics")

            # Metrics cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.4f}", help="Proporsi prediksi yang benar")
            m2.metric("Precision", f"{prec:.4f}", help="Ketepatan prediksi positif")
            m3.metric("Recall", f"{rec:.4f}", help="Kemampuan mendeteksi kelas positif")
            m4.metric("F1 Score", f"{f1:.4f}", help="Harmonic mean dari Precision & Recall")

            # Penjelasan metrics
            st.markdown("---")
            st.subheader("📝 Penjelasan Metrics")
            st.markdown(f"""
            | Metric | Nilai | Keterangan |
            |--------|-------|------------|
            | **Accuracy** | {acc:.4f} | Persentase prediksi yang benar dari total data. |
            | **Precision** | {prec:.4f} | Dari yang diprediksi positif, berapa yang benar. |
            | **Recall** | {rec:.4f} | Dari yang sebenarnya positif, berapa yang terdeteksi. |
            | **F1 Score** | {f1:.4f} | Keseimbangan antara Precision dan Recall. |
            """)

            # Interpretasi otomatis
            st.subheader("🔍 Interpretasi")
            if acc >= 0.95:
                st.success(f"🌟 Model **sangat baik**! Accuracy = {acc:.4f} (≥ 95%)")
            elif acc >= 0.85:
                st.info(f"👍 Model **baik**. Accuracy = {acc:.4f} (≥ 85%)")
            elif acc >= 0.70:
                st.warning(f"⚠️ Model **cukup**. Accuracy = {acc:.4f} (≥ 70%). Coba model lain.")
            else:
                st.error(f"❌ Model **kurang baik**. Accuracy = {acc:.4f} (< 70%). Ganti model atau dataset.")

            # Classification Report
            st.subheader("📋 Classification Report")
            st.code(report, language="text")

            # Bar chart metrics
            st.subheader("📊 Perbandingan Metrics")
            fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
            metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
            metric_values = [acc, prec, rec, f1]
            colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
            bars = ax_bar.bar(metric_names, metric_values, color=colors, edgecolor="black")
            for bar, val in zip(bars, metric_values):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold",
                )
            ax_bar.set_ylabel("Nilai")
            ax_bar.set_ylim(0, 1.15)
            ax_bar.set_title("Perbandingan Classification Metrics")
            ax_bar.grid(True, alpha=0.3, axis="y")
            st.pyplot(fig_bar)

# ===================== CLUSTERING =====================
elif jenis_task == "Clustering":
    st.title("🔮 Clustering Machine Learning")
    st.write("Aplikasi demo Clustering (Unsupervised Learning) menggunakan dataset dari scikit-learn atau MongoDB.")

    st.info("👈 Silakan atur parameter di sidebar, lalu klik **🚀 Train Model** untuk memulai clustering.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Dataset")
        st.write("Scikit-learn atau MongoDB")
    with col2:
        st.markdown("### 🧠 Algoritma")
        st.write("K-Means, DBSCAN, Agglomerative")
    with col3:
        st.markdown("### 📐 Evaluasi")
        st.write("Silhouette Score + Elbow Method")

    if tombol_train:
        # Muat data berdasarkan sumber
        if sumber_data_clust == "Scikit-learn":
            df, feature_names = load_clustering_data(dataset_clustering)
            X = df[feature_names]
            nama_dataset_tampil = dataset_clustering
        else:
            if dataset_clustering_mongo is None:
                st.error("❌ Belum ada dataset clustering di MongoDB!")
                st.stop()
            doc = mongo_read_dataset_full(dataset_clustering_mongo)
            df = pd.DataFrame(doc["data"])
            # Semua kolom numerik sebagai fitur
            feature_names = [c for c in doc["columns"] if df[c].dtype in ["int64", "float64"]]
            X = df[feature_names]
            nama_dataset_tampil = f"{dataset_clustering_mongo} (MongoDB)"

        # Scaling data (penting untuk clustering)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = get_clustering_model(model_clustering, n_clusters, eps_val, min_samples_val)
        start_time = time.time()
        labels = model.fit_predict(X_scaled)
        training_time = time.time() - start_time

        # Jumlah cluster yang ditemukan
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Hitung Silhouette Score (hanya jika ada >= 2 cluster dan tidak semua noise)
        if n_clusters_found >= 2:
            sil_score = silhouette_score(X_scaled, labels)
        else:
            sil_score = None

        # PCA untuk visualisasi 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_.sum() * 100

        # Success message
        st.success(f"✅ Clustering selesai dalam {training_time:.3f} detik!")

        # Info ringkasan
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Task", "Clustering")
        col2.metric("📊 Dataset", nama_dataset_tampil)
        col3.metric("🧠 Algoritma", model_clustering.split()[0])
        col4.metric("⏱️ Training Time", f"{training_time:.3f}s")

        # --- Tabs ---
        tab1, tab2 = st.tabs(["📊 Visualisasi Cluster", "📐 Evaluation Metrics"])

        # ===== TAB 1: Visualisasi =====
        with tab1:
            st.header("📊 Hasil Visualisasi Cluster")

            # Info data
            c1, c2, c3 = st.columns(3)
            c1.success(f"Total Data: {len(X)} sampel")
            c2.info(f"Cluster Ditemukan: {n_clusters_found}")
            if n_noise > 0:
                c3.warning(f"Noise Points: {n_noise}")
            else:
                c3.success(f"Noise Points: {n_noise}")

            # Scatter plot PCA 2D
            st.subheader(f"🗺️ Scatter Plot (PCA 2D — {explained_var:.1f}% variance explained)")
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 7))

            # Warna untuk setiap cluster
            unique_labels = sorted(set(labels))
            colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

            for i, label in enumerate(unique_labels):
                mask = labels == label
                if label == -1:
                    # Noise points (DBSCAN)
                    ax_scatter.scatter(
                        X_pca[mask, 0], X_pca[mask, 1],
                        c="gray", marker="x", s=50, alpha=0.5,
                        label="Noise",
                    )
                else:
                    ax_scatter.scatter(
                        X_pca[mask, 0], X_pca[mask, 1],
                        c=[colors_list[i % len(colors_list)]], marker="o", s=60, alpha=0.7,
                        edgecolors="k", linewidths=0.5,
                        label=f"Cluster {label}",
                    )

            ax_scatter.set_xlabel("PC1", fontsize=12)
            ax_scatter.set_ylabel("PC2", fontsize=12)
            ax_scatter.set_title(f"Hasil Clustering — {model_clustering}", fontsize=14)
            ax_scatter.legend(loc="best")
            ax_scatter.grid(True, alpha=0.3)
            st.pyplot(fig_scatter)

            # Distribusi per cluster
            st.subheader("📊 Distribusi Anggota per Cluster")
            fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            colors_bar = [colors_list[i % len(colors_list)] if label != -1 else "gray"
                          for i, label in enumerate(sorted(cluster_counts.index))]
            bar_labels = [f"Cluster {l}" if l != -1 else "Noise" for l in sorted(cluster_counts.index)]
            bars = ax_dist.bar(bar_labels, cluster_counts.values, color=colors_bar, edgecolor="black")
            for bar, val in zip(bars, cluster_counts.values):
                ax_dist.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    str(val), ha="center", va="bottom", fontweight="bold",
                )
            ax_dist.set_ylabel("Jumlah Anggota")
            ax_dist.set_title("Jumlah Anggota per Cluster")
            ax_dist.grid(True, alpha=0.3, axis="y")
            st.pyplot(fig_dist)

        # ===== TAB 2: Evaluation =====
        with tab2:
            st.header("📐 Evaluation Metrics")

            # Silhouette Score
            m1, m2, m3 = st.columns(3)
            m1.metric("🔢 Jumlah Cluster", n_clusters_found)
            m2.metric(
                "📏 Silhouette Score",
                f"{sil_score:.4f}" if sil_score is not None else "N/A",
                help="Skor -1 s/d 1. Semakin tinggi semakin baik.",
            )
            m3.metric("🔊 Noise Points", n_noise)

            # Penjelasan
            st.markdown("---")
            st.subheader("📝 Penjelasan Metrics")
            st.markdown("""
            | Metric | Keterangan |
            |--------|------------|
            | **Silhouette Score** | Mengukur seberapa baik objek berada di cluster-nya dibandingkan cluster tetangga. Range: -1 s/d 1. |
            | **Jumlah Cluster** | Total cluster yang ditemukan oleh algoritma. |
            | **Noise Points** | Titik data yang tidak masuk cluster manapun (khusus DBSCAN). |
            """)

            # Interpretasi
            st.subheader("🔍 Interpretasi")
            if sil_score is not None:
                if sil_score >= 0.7:
                    st.success(f"🌟 Clustering **sangat baik**! Silhouette = {sil_score:.4f} (≥ 0.7)")
                elif sil_score >= 0.5:
                    st.info(f"👍 Clustering **baik**. Silhouette = {sil_score:.4f} (≥ 0.5)")
                elif sil_score >= 0.25:
                    st.warning(f"⚠️ Clustering **cukup**. Silhouette = {sil_score:.4f} (≥ 0.25). Coba ubah parameter.")
                else:
                    st.error(f"❌ Clustering **kurang baik**. Silhouette = {sil_score:.4f} (< 0.25). Coba algoritma/parameter lain.")
            else:
                st.warning("⚠️ Silhouette Score tidak bisa dihitung (cluster < 2).")

            # Elbow Method (khusus K-Means)
            if model_clustering == "K-Means":
                st.subheader("📉 Elbow Method")
                st.write("Grafik ini membantu menentukan jumlah cluster optimal.")

                inertias = []
                sil_scores = []
                K_range = range(2, 11)
                for k in K_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    inertias.append(km.inertia_)
                    sil_scores.append(silhouette_score(X_scaled, km.labels_))

                fig_elbow, (ax_e1, ax_e2) = plt.subplots(1, 2, figsize=(14, 5))

                # Elbow (Inertia)
                ax_e1.plot(list(K_range), inertias, "bo-", linewidth=2, markersize=8)
                ax_e1.axvline(x=n_clusters, color="red", linestyle="--", alpha=0.7, label=f"K = {n_clusters}")
                ax_e1.set_xlabel("Jumlah Cluster (K)")
                ax_e1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
                ax_e1.set_title("Elbow Method")
                ax_e1.legend()
                ax_e1.grid(True, alpha=0.3)

                # Silhouette per K
                ax_e2.plot(list(K_range), sil_scores, "go-", linewidth=2, markersize=8)
                ax_e2.axvline(x=n_clusters, color="red", linestyle="--", alpha=0.7, label=f"K = {n_clusters}")
                ax_e2.set_xlabel("Jumlah Cluster (K)")
                ax_e2.set_ylabel("Silhouette Score")
                ax_e2.set_title("Silhouette Score per K")
                ax_e2.legend()
                ax_e2.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig_elbow)

                # Tabel K vs Score
                st.subheader("📋 Tabel Perbandingan K")
                df_elbow = pd.DataFrame({
                    "K": list(K_range),
                    "Inertia": [f"{v:.2f}" for v in inertias],
                    "Silhouette Score": [f"{v:.4f}" for v in sil_scores],
                })
                best_k = list(K_range)[np.argmax(sil_scores)]
                st.dataframe(df_elbow, use_container_width=True, hide_index=True)
                st.info(f"💡 Berdasarkan Silhouette Score, jumlah cluster terbaik adalah **K = {best_k}**.")

# ===================== MANAJEMEN DATASET (MongoDB) =====================
elif jenis_task == "Manajemen Dataset":
    st.title("🗄️ Manajemen Dataset (MongoDB)")
    st.write("Upload, lihat, update, dan hapus dataset di MongoDB Atlas.")

    # Test koneksi
    db = get_database()
    if db is not None:
        st.success("✅ Terhubung ke MongoDB Atlas!")
    else:
        st.error("❌ Gagal terhubung ke MongoDB. Periksa koneksi dan credentials.")
        st.stop()

    # ===== UPLOAD DATASET =====
    if operasi_crud == "📤 Upload Dataset":
        st.header("📤 Upload Dataset Baru")
        st.write("Upload file CSV untuk disimpan ke MongoDB.")

        with st.form("form_upload"):
            nama_dataset = st.text_input(
                "📝 Nama Dataset",
                placeholder="Contoh: Data Penjualan 2024",
            )
            deskripsi = st.text_area(
                "📋 Deskripsi",
                placeholder="Deskripsi singkat tentang dataset ini...",
            )
            tipe_dataset = st.selectbox(
                "🎯 Tipe Dataset",
                ["Regresi", "Klasifikasi", "Clustering"],
                help="Pilih tipe sesuai task yang akan digunakan",
            )
            uploaded_file = st.file_uploader(
                "📁 Upload File CSV",
                type=["csv"],
            )

            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.info("💡 **Untuk Regresi & Klasifikasi:** kolom terakhir = target/label")
            with col_info2:
                st.info("💡 **Untuk Clustering:** semua kolom numerik = fitur")

            submitted = st.form_submit_button("💾 Simpan ke MongoDB", use_container_width=True, type="primary")

            if submitted:
                if not nama_dataset:
                    st.error("❌ Nama dataset harus diisi!")
                elif uploaded_file is None:
                    st.error("❌ Upload file CSV terlebih dahulu!")
                else:
                    try:
                        df_upload = pd.read_csv(uploaded_file)
                        st.write("**Preview data (5 baris pertama):**")
                        st.dataframe(df_upload.head(), use_container_width=True)

                        success, msg = mongo_create_dataset(nama_dataset, deskripsi, tipe_dataset, df_upload)
                        if success:
                            st.success(f"✅ {msg}")
                            st.balloons()
                        else:
                            st.error(f"❌ {msg}")
                    except Exception as e:
                        st.error(f"❌ Error membaca CSV: {e}")

    # ===== LIHAT SEMUA DATASET =====
    elif operasi_crud == "📋 Lihat Semua Dataset":
        st.header("📋 Daftar Dataset di MongoDB")

        datasets_list = mongo_read_datasets()

        if not datasets_list:
            st.warning("📭 Belum ada dataset di MongoDB. Upload dataset terlebih dahulu!")
        else:
            st.info(f"📊 Total: **{len(datasets_list)}** dataset tersimpan")

            for i, ds in enumerate(datasets_list):
                with st.expander(f"📁 {ds['nama']} — {ds.get('tipe', 'N/A')} ({ds.get('jumlah_baris', '?')} baris)", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📊 Baris", ds.get("jumlah_baris", "?"))
                    col2.metric("📐 Kolom", ds.get("jumlah_kolom", "?"))
                    col3.metric("🎯 Tipe", ds.get("tipe", "?"))

                    st.write(f"**Deskripsi:** {ds.get('deskripsi', '-')}")
                    st.write(f"**Kolom:** {', '.join(ds.get('columns', []))}")
                    st.write(f"**Dibuat:** {ds.get('created_at', '-')}")

                    # Tombol lihat detail
                    if st.button(f"🔍 Lihat Data", key=f"view_{i}"):
                        doc = mongo_read_dataset_full(ds["nama"])
                        if doc:
                            df_view = pd.DataFrame(doc["data"])
                            st.dataframe(df_view, use_container_width=True)
                            st.write(f"**Shape:** {df_view.shape}")
                            st.write(f"**Tipe data:**")
                            st.write(df_view.dtypes)

    # ===== UPDATE DATASET =====
    elif operasi_crud == "✏️ Update Dataset":
        st.header("✏️ Update Info Dataset")

        datasets_list = mongo_read_datasets()
        if not datasets_list:
            st.warning("📭 Belum ada dataset di MongoDB.")
        else:
            nama_list = [ds["nama"] for ds in datasets_list]
            selected = st.selectbox("📁 Pilih Dataset", nama_list)

            # Ambil info saat ini
            ds_info = next((ds for ds in datasets_list if ds["nama"] == selected), None)

            if ds_info:
                with st.form("form_update"):
                    nama_baru = st.text_input("📝 Nama Baru", value=ds_info["nama"])
                    deskripsi_baru = st.text_area(
                        "📋 Deskripsi Baru",
                        value=ds_info.get("deskripsi", ""),
                    )

                    submitted = st.form_submit_button("💾 Update", use_container_width=True, type="primary")

                    if submitted:
                        success, msg = mongo_update_dataset(selected, nama_baru, deskripsi_baru)
                        if success:
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")

    # ===== HAPUS DATASET =====
    elif operasi_crud == "🗑️ Hapus Dataset":
        st.header("🗑️ Hapus Dataset")

        datasets_list = mongo_read_datasets()
        if not datasets_list:
            st.warning("📭 Belum ada dataset di MongoDB.")
        else:
            nama_list = [ds["nama"] for ds in datasets_list]
            selected = st.selectbox("📁 Pilih Dataset yang akan dihapus", nama_list)

            ds_info = next((ds for ds in datasets_list if ds["nama"] == selected), None)
            if ds_info:
                st.warning(f"⚠️ Anda akan menghapus dataset **{selected}** ({ds_info.get('jumlah_baris', '?')} baris)")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Ya, Hapus!", type="primary", use_container_width=True):
                        success, msg = mongo_delete_dataset(selected)
                        if success:
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
                with col2:
                    if st.button("❌ Batal", use_container_width=True):
                        st.info("Penghapusan dibatalkan.")

# ===================== KOMENTAR PUBLIK =====================
elif jenis_task == "Komentar Publik":
    st.title("💬 Komentar Publik")
    st.write("Tinggalkan komentar, saran, atau pertanyaan tentang aplikasi ini.")

    # Test koneksi
    db = get_database()
    if db is not None:
        st.success("✅ Terhubung ke MongoDB Atlas!")
    else:
        st.error("❌ Gagal terhubung ke MongoDB.")
        st.stop()

    # --- Form komentar ---
    st.subheader("✍️ Tulis Komentar")
    with st.form("form_komentar", clear_on_submit=True):
        nama_user = st.text_input(
            "👤 Nama",
            placeholder="Masukkan nama kamu...",
        )
        isi_komentar = st.text_area(
            "💬 Komentar",
            placeholder="Tulis komentar, saran, atau pertanyaan...",
        )
        submit_komentar = st.form_submit_button(
            "📩 Kirim Komentar", use_container_width=True, type="primary",
        )

        if submit_komentar:
            if not nama_user.strip():
                st.error("❌ Nama harus diisi!")
            elif not isi_komentar.strip():
                st.error("❌ Komentar harus diisi!")
            else:
                success, msg = mongo_add_comment(nama_user.strip(), isi_komentar.strip())
                if success:
                    st.success(f"✅ {msg}")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

    # --- Daftar komentar ---
    st.markdown("---")
    st.subheader("📋 Semua Komentar")

    comments = mongo_get_comments()

    if not comments:
        st.info("📭 Belum ada komentar. Jadilah yang pertama!")
    else:
        st.caption(f"Total: {len(comments)} komentar")

        for i, comment in enumerate(comments):
            with st.container():
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(
                        f"**👤 {comment['nama']}** — "
                        f"<small style='color: gray;'>{comment.get('waktu', '-')}</small>",
                        unsafe_allow_html=True,
                    )
                    st.write(comment["komentar"])
                with col2:
                    if st.button("🗑️", key=f"del_comment_{i}", help="Hapus komentar"):
                        mongo_delete_comment(str(comment["_id"]))
                        st.rerun()
                st.markdown("---")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🤖 TAS Mata Kuliah AI — UKSW 2026"
    "</div>",
    unsafe_allow_html=True,
)
