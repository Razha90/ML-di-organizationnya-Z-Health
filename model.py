import pickle
import numpy as np
import random
import json
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

with open("./database/recommended_activities.json", "r") as file:
    recommended_activities = json.load(file)
model = load_model('./model/model_stress_level.keras')
scaler = pickle.load(open('./model/scaler.pkl', 'rb'))
tokenizer = pickle.load(open('./model/tokenizer.pkl', 'rb'))

    
def get_recommendations(stress_level, journal_text):
    matched_categories = match_keywords_in_journal(journal_text)

    recommendations = []
    for category in matched_categories:
        for activity in recommended_activities.get(stress_level, []):
            if category.lower() in activity['icon'].lower():
                recommendations.append(activity)

    if not recommendations:
        recommendations = random.sample(recommended_activities.get(stress_level, []), 1)

    return recommendations
  
keyword_categories = {
    "konseling": ["trauma", "konseling", "pengacara", "mentor"],
    "hotline": ["layanan bantuan", "hotline", "korban", "layanan darurat", "polisi", "pihak berwenang", "rencana keselamatan"],
    "sosialisasi": ["jaringan", "dukungan sosial", "koneksi sosial", "bersosialisasi", "aktivitas sosial"],
    "olahraga": ["kesehatan fisik", "fisik", "aktivitas fisik", "olahraga"],
    "tidur": ["istirahat", "cukup tidur", "tidur", "pola tidur"],
    "pola makan": ["makan sehat", "bergizi", "camilan", "makanan sehat"],
    "meditasi": ["relaksasi", "yoga", "manajemen stress", "rutinitas pagi", "teknik manajemen stress", "mekanisme koping", "terapi"],
    "Kelola Waktu": ["jadwal", "waktu efektif", "tugas"],
    "menggambar": ["menggambar", "kreativitas"],
    "hobi": ["hobi", "aktivitas yang anda nikmati", "kesenangan"],
    "media sosial": ["media sosial"],
    "kafein dan alcohol": ["kafein", "alkohol"],
    "afirmasi diri": ["pujian", "afirmasi", "kepercayaan diri"],
    "atur keuangan": ["keuangan"],
    "bantuan professional": ["terapis", "profesional"],
    "komunikasi": ["pasangan", "hubungan", "komunikasi", "komunikasikan", "manajemen", "suami"],
    "pengobatan": ["pengobatan", "obat"],
    "perawatan diri": ["perawatan diri", "terima bentuk tubuh"],
    "batasi jam kerja": ["jam kerja"],
    "hubungi teman": ["teman", "kawan", "sahabat"],
    "hubungi orang tua": ["dukungan", "orang tua", "keluarga", "kerabat"],
    "atur emosi": ["fakta emosi", "emosi", "fokus hal positif", "situasi secara objektif", "hal-hal yang dapat dikendalikan", "pikiran negatif"],
    "identifikasi": ["identifikasi"]
}


def process_journal(journal_text):
    # Tokenisasi dan penghapusan stopwords
    stop_words = set(stopwords.words('indonesian'))
    words = word_tokenize(journal_text.lower())  # Ubah menjadi lowercase dan tokenisasi
    filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
    return filtered_words

# Fungsi untuk mencari kecocokan kata kunci dalam jurnal harian
def match_keywords_in_journal(journal_text):
    processed_words = process_journal(journal_text)
    matched_categories = []

    # Cek apakah ada kata kunci dari jurnal yang cocok dengan kategori
    for category, keywords in keyword_categories.items():
        if any(keyword in processed_words for keyword in keywords):
            matched_categories.append(category)

    return matched_categories

def getAllRecommended(waktu_belajar, waktu_belajar_tambahan, waktu_tidur, aktivitas_sosial, aktivitas_fisik, jurnal_harian):
    fitur_numeric = np.array([[waktu_belajar,waktu_belajar_tambahan,waktu_tidur,aktivitas_sosial,aktivitas_fisik]])
    fitur_numeric_scaler = scaler.transform(fitur_numeric)
    
    fitur_text = jurnal_harian
    fitur_text_tokenized = tokenizer.texts_to_sequences([fitur_text])
    fitur_text_token = pad_sequences(fitur_text_tokenized, maxlen=100, padding='post')

    fitur_numeric_scaled = np.array(fitur_numeric_scaler)
    fitur_text_padded = np.array(fitur_text_token)

    # Prediksi tingkat stres
    stress_prediction = model.predict([fitur_numeric_scaled, fitur_text_padded])
    stress_level = np.argmax(stress_prediction)

    stress_mapping = {0: 'High', 1: 'Low', 2: 'Moderate'}
    predicted_stress = stress_mapping[stress_level]
    # print(f"Tingkat Stres: {predicted_stress}")

    recommendations = get_recommendations(predicted_stress, jurnal_harian)
    
    return {
        "predicted_stress": predicted_stress,
        "recommendations": recommendations
    }
