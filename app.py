import streamlit as st
import pandas as pd
import joblib
import os

# === HARUS PALING AWAL ===
st.set_page_config(page_title="Prediksi Sentimen Manual", layout="centered")

# === Load model ===
@st.cache_resource
def load_model():
    return joblib.load("naive_bayes_model.pkl")

model = load_model()

# === Judul Aplikasi ===
st.title("🧠 Sistem Analisis Sentiment Masyarakat Terhadap Pelayanan Puskesmar Di Kendari (Input Manual)")
st.write("Ketik kalimat secara manual, lalu klik tombol prediksi. Hasil akan otomatis disimpan ke file CSV.")

# === Input user ===
user_input = st.text_area("📝 Masukkan kalimat Anda di sini:", height=150)

if st.button("🔍 Prediksi Sentimen"):
    if not user_input.strip():
        st.warning("⚠️ Kalimat tidak boleh kosong.")
    else:
        # Prediksi
        prediction = model.predict([user_input])[0]
        proba = model.predict_proba([user_input])[0]
        label_map = {0: "Negatif", 1: "Positif", 2: "Netral"}
        color_map = {0: "#ff4d4d", 1: "#4CAF50", 2: "#2196F3"}

        predicted_label = label_map.get(prediction, str(prediction))
        confidence = round(100 * max(proba), 2)
        color = color_map.get(prediction, "#333333")

        # Tampilkan hasil dengan warna
        st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {color}; color: white; text-align: center;'>
                <h3>✅ Hasil Prediksi: {predicted_label}</h3>
                <p>📊 Confidence: <strong>{confidence}%</strong></p>
            </div>
        """, unsafe_allow_html=True)

        # Simpan ke CSV
        output_file = "hasil_prediksi_manual.csv"
        new_row = pd.DataFrame([[user_input, predicted_label]], columns=["Kalimat", "Prediksi"])

        if os.path.exists(output_file):
            old_df = pd.read_csv(output_file)
            combined_df = pd.concat([old_df, new_row], ignore_index=True)
        else:
            combined_df = new_row

        combined_df.to_csv(output_file, index=False)
        st.info(f"Hasil prediksi disimpan ke **{output_file}**")

        # Tampilkan semua hasil
        with st.expander("📂 Lihat Semua Hasil yang Tersimpan"):
            st.dataframe(combined_df)
