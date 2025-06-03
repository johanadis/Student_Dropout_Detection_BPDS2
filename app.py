import streamlit as st
import pandas as pd
import pickle

# Mapping nilai numerik ke label deskripsi
MAPPING_DICT = {
    'Status Pernikahan':  {
        0: 'Lajang',
        1: 'Menikah',
        2: 'Janda/Duda',
        3: 'Bercerai',
        4: 'Hidup Bersama',
        5: 'Pisah Secara Hukum'
    },
    'Kehadiran Siang/Malam': {
        1: 'Siang',
        0: 'Malam'
    },
    'Jenis Kelamin': {
        1: 'Laki-laki',
        0: 'Perempuan'
    },
    'Terdampak': {
        1: 'Ya',
        0: 'Tidak'
    },
    'Kebutuhan Khusus': {
        1: 'Ya',
        0: 'Tidak'
    },
    'Memiliki Hutang': {
        1: 'Ya',
        0: 'Tidak'
    },
    'SPP Terbayar': {
        1: 'Ya',
        0: 'Tidak'
    },
    'Penerima Beasiswa': {
        1: 'Ya',
        0: 'Tidak'
    },
    'Mahasiswa Internasional': {
        1: 'Ya',
        0: 'Tidak'
    },
    'Mahasiswa Lokal': {
        1: 'Ya',
        0: 'Tidak'
    },
    'Kedua Orang Tua Bekerja': {
        1: 'Ya',
        0: 'Tidak'
    }
}

# Load model, scaler, dan daftar fitur
with open('model/best_model_randomforest.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="centered")
st.title("🎓 Prediksi Dropout Mahasiswa")
st.markdown(
    "Silakan lengkapi data mahasiswa berikut untuk memprediksi kemungkinan dropout.")

# Formulir input data mahasiswa
with st.form("dropout_form"):
    col1, col2 = st.columns(2)

    with col1:
        marital_status = st.selectbox(
            "Status Pernikahan",
            list(MAPPING_DICT['Status Pernikahan'].keys()),
            format_func=lambda x: MAPPING_DICT['Status Pernikahan'][x],
            help="Pilih status pernikahan mahasiswa"
        )
        application_mode = st.number_input("Mode Aplikasi", 0, 20, 1)
        application_order = st.number_input("Urutan Aplikasi", 0, 10, 1)
        attendance = st.selectbox(
            "Kehadiran Siang/Malam", [0, 1], format_func=lambda x: MAPPING_DICT['Kehadiran Siang/Malam'][x],
            help="Pilih waktu kehadiran mahasiswa"
        )
        prev_qualification = st.number_input(
            "Kualifikasi Sebelumnya", 0, 20, 1)
        prev_grade = st.number_input(
            "Nilai Kualifikasi Sebelumnya", 0.0, 200.0, 150.0)
        admission_grade = st.number_input("Nilai Masuk", 0.0, 200.0, 150.0)
        displaced = st.selectbox(
            "Terdampak", [0, 1], format_func=lambda x: MAPPING_DICT['Terdampak'][x],
            help="Apakah mahasiswa terdampak?"
        )
        special_needs = st.selectbox(
            "Kebutuhan Khusus", [0, 1], format_func=lambda x: MAPPING_DICT['Kebutuhan Khusus'][x],
            help="Apakah mahasiswa memiliki kebutuhan khusus?"
        )
        debtor = st.selectbox(
            "Memiliki Hutang", [0, 1], format_func=lambda x: MAPPING_DICT['Memiliki Hutang'][x],
            help="Apakah mahasiswa memiliki hutang?"
        )
        tuition_paid = st.selectbox(
            "SPP Terbayar", [0, 1], format_func=lambda x: MAPPING_DICT['SPP Terbayar'][x],
            help="Apakah SPP mahasiswa sudah terbayar?"
        )
        gender = st.selectbox(
            "Jenis Kelamin", [0, 1], format_func=lambda x: MAPPING_DICT['Jenis Kelamin'][x],
            help="Pilih jenis kelamin mahasiswa"
        )
        scholarship_holder = st.selectbox(
            "Penerima Beasiswa", [0, 1], format_func=lambda x: MAPPING_DICT['Penerima Beasiswa'][x],
            help="Apakah mahasiswa penerima beasiswa?"
        )
        age = st.number_input("Usia Saat Masuk", 15, 100, 20)
        international = st.selectbox(
            "Mahasiswa Internasional", [0, 1], format_func=lambda x: MAPPING_DICT['Mahasiswa Internasional'][x],
            help="Apakah mahasiswa internasional?"
        )
        sem1_enrolled = st.number_input(
            "Mata Kuliah Semester 1 Diambil", 0, 20, 6)
        sem1_evals = st.number_input("Evaluasi Semester 1", 0, 20, 6)
        sem1_approved = st.number_input("Lulus Semester 1", 0, 20, 6)
        sem1_grade = st.number_input("Nilai Semester 1", 0.0, 20.0, 14.0)

    with col2:
        sem1_credited = st.number_input("SKS Semester 1", 0, 20, 0)
        sem1_wo_eval = st.number_input("Tanpa Evaluasi Semester 1", 0, 10, 0)
        sem2_enrolled = st.number_input(
            "Mata Kuliah Semester 2 Diambil", 0, 20, 6)
        sem2_evals = st.number_input("Evaluasi Semester 2", 0, 20, 6)
        sem2_approved = st.number_input("Lulus Semester 2", 0, 20, 6)
        sem2_grade = st.number_input("Nilai Semester 2", 0.0, 20.0, 14.0)
        sem2_credited = st.number_input("SKS Semester 2", 0, 20, 0)
        sem2_wo_eval = st.number_input("Tanpa Evaluasi Semester 2", 0, 10, 0)
        unemployment = st.number_input(
            "Tingkat Pengangguran (%)", 0.0, 100.0, 6.5)
        inflation = st.number_input("Tingkat Inflasi (%)", -10.0, 100.0, 1.2)
        gdp = st.number_input("GDP", 0.0, 1000.0, 180.0)
        course_group = st.number_input("Kelompok Program Studi", 0, 10, 1)
        is_local = st.selectbox(
            "Mahasiswa Lokal", [0, 1], format_func=lambda x: MAPPING_DICT['Mahasiswa Lokal'][x],
            help="Apakah mahasiswa lokal?"
        )
        mother_edu = st.number_input("Tingkat Pendidikan Ibu", 0, 10, 3)
        father_edu = st.number_input("Tingkat Pendidikan Ayah", 0, 10, 4)
        edu_gap = st.number_input("Gap Pendidikan Orang Tua", -10, 10, -1)
        mother_job = st.number_input("Pekerjaan Ibu", 0, 20, 10)
        father_job = st.number_input("Pekerjaan Ayah", 0, 20, 10)
        parents_work = st.selectbox(
            "Kedua Orang Tua Bekerja", [0, 1], format_func=lambda x: MAPPING_DICT['Kedua Orang Tua Bekerja'][x],
            help="Apakah kedua orang tua bekerja?"
        )

    submitted = st.form_submit_button("Prediksi")

    if submitted:
        # Membuat dictionary input sesuai urutan fitur model
        input_data = {
            "Marital_status": marital_status,
            "Application_mode": application_mode,
            "Application_order": application_order,
            "Daytime_evening_attendance": attendance,
            "Previous_qualification": prev_qualification,
            "Previous_qualification_grade": prev_grade,
            "Admission_grade": admission_grade,
            "Displaced": displaced,
            "Educational_special_needs": special_needs,
            "Debtor": debtor,
            "Tuition_fees_up_to_date": tuition_paid,
            "Gender": gender,
            "Scholarship_holder": scholarship_holder,
            "Age_at_enrollment": age,
            "International": international,
            "Curricular_units_1st_sem_enrolled": sem1_enrolled,
            "Curricular_units_1st_sem_evaluations": sem1_evals,
            "Curricular_units_1st_sem_approved": sem1_approved,
            "Curricular_units_1st_sem_grade": sem1_grade,
            "Curricular_units_1st_sem_credited": sem1_credited,
            "Curricular_units_1st_sem_without_evaluations": sem1_wo_eval,
            "Curricular_units_2nd_sem_enrolled": sem2_enrolled,
            "Curricular_units_2nd_sem_evaluations": sem2_evals,
            "Curricular_units_2nd_sem_approved": sem2_approved,
            "Curricular_units_2nd_sem_grade": sem2_grade,
            "Curricular_units_2nd_sem_credited": sem2_credited,
            "Curricular_units_2nd_sem_without_evaluations": sem2_wo_eval,
            "Unemployment_rate": unemployment,
            "Inflation_rate": inflation,
            "GDP": gdp,
            "Course_group": course_group,
            "Is_local": is_local,
            "Mother_edu_level": mother_edu,
            "Father_edu_level": father_edu,
            "Mother_job": mother_job,
            "Father_job": father_job,
            "Parental_education_gap": edu_gap,
            "Is_both_parents_employed": parents_work
        }

        # Proses prediksi
        df_input = pd.DataFrame([input_data])
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)
        scaled = scaler.transform(df_input)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        result = "🔴 Dropout" if prediction == 1 else "🟢 Tidak Dropout"
        st.success(f"🎯 Hasil Prediksi Status Mahasiswa: **{result}**")
        st.metric("⚠️ Probabilitas Dropout", f"{probability:.2%}")
