import pandas as pd
import pickle

""" 
predict.py 
Kode Sederhana Untuk Melakukan Prediksi
Menggunakan Model Random Forest yang Sudah Dilatih
"""

#  Load model, scaler, dan feature columns
with open('model/best_model_randomforest.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)


def predict(data_dict):
    # Buat DataFrame dari input dictionary
    df_input = pd.DataFrame([data_dict])

    # Reorder kolom dan isi yang kurang dengan 0
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    # Scaling
    df_scaled = scaler.transform(df_input)

    # Prediksi
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    label = "Dropout" if prediction == 1 else "Tidak Dropout"
    return label, probability


if __name__ == "__main__":
    # Contoh input ditulis secara manual
    example_input = {
        'Marital_status': 1,
        'Application_mode': 7,
        'Application_order': 1,
        'Daytime_evening_attendance': 1,
        'Previous_qualification': 1,
        'Previous_qualification_grade': 135.0,
        'Admission_grade': 140.0,
        'Displaced': 1,
        'Educational_special_needs': 0,
        'Debtor': 1,
        'Tuition_fees_up_to_date': 1,
        'Gender': 0,
        'Scholarship_holder': 0,
        'Age_at_enrollment': 20,
        'International': 0,
        'Curricular_units_1st_sem_credited': 0,
        'Curricular_units_1st_sem_enrolled': 6,
        'Curricular_units_1st_sem_evaluations': 6,
        'Curricular_units_1st_sem_approved': 6,
        'Curricular_units_1st_sem_grade': 11,
        'Curricular_units_1st_sem_without_evaluations': 0,
        'Curricular_units_2nd_sem_credited': 0,
        'Curricular_units_2nd_sem_enrolled': 6,
        'Curricular_units_2nd_sem_evaluations': 6,
        'Curricular_units_2nd_sem_approved': 6,
        'Curricular_units_2nd_sem_grade': 13,
        'Curricular_units_2nd_sem_without_evaluations': 0,
        'Unemployment_rate': 6.9,
        'Inflation_rate': 1.3,
        'GDP': 170.0,
        'Course_group': 1,
        'Is_local': 1,
        'Mother_edu_level': 3,
        'Father_edu_level': 4,
        'Mother_job': 7,
        'Father_job': 9,
        'Parental_education_gap': -1,
        'Is_both_parents_employed': 1
    }

    # Jalankan prediksi
    label, prob = predict(example_input)

    print(f"\n🎯 Hasil Prediksi Status Mahasiswa: {label}")
    print(f"⚠️  Probabilitas  Dropout: {prob:.2%}\n")
