import pandas as pd
import streamlit as st
import joblib


st.set_page_config(
	page_title = "Klasifikasi Kesehatan Mental",
	page_icon = ":medical:"
)

model = joblib.load("model_klasifikasi_kesehatan_mental.joblib")

st.title("Klasifikasi Kesehatan Mental")
st.markdown("klasifikasi kesehatan mental anak berdasarkan data")

Jurusan = st.pills("Jurusan",["Teknik Mesin","Akuntansi", "Multimedia" ,"Perkantoran", "Teknik Otomotif","Tata Boga"],default ="Akuntansi"  )
Usia	= st.slider("Usia", 14, 19, 15)
JenisKelamin	= st.pills("Jenis Kelamin",["Laki-Laki","Perempuan"],default = "Perempuan")
PendapatanKeluarga	= st.pills("Pendapatan Keluarga",["Rendah","Menengah","Tinggi"],default = "Rendah")	
LokasiSekolah	= st.pills("Lokasi Sekolah",["Urban","Rural","Suburban"],default = "Rural")
JumlahJamHPHarian = st.slider("Jumlah Jam HP Harian", 1, 12,9)


if st.button("Potensi Burnout", type="primary"):
	data_baru = pd.DataFrame([[Jurusan,Usia,JenisKelamin,PendapatanKeluarga,LokasiSekolah,JumlahJamHPHarian]], columns=["Jurusan","Usia","Jenis Kelamin","Pendapatan Keluarga","Lokasi Sekolah","Jumlah Jam HP Harian"])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"prediksi {prediksi} dan presentase {presentase*100:.2f}%")
	st.snow()
st.divider()
st.caption("diuat oleh silfi")