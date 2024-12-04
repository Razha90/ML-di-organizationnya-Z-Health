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


with open("./database/score_data.json", "r") as file:
    my_data = json.load(file)

with open("./database/recommended_activities.json", "r") as file:
    recommended_activities = json.load(file)
model = load_model('./model/model_stress_level.keras')
scaler = pickle.load(open('./model/scaler.pkl', 'rb'))
tokenizer = pickle.load(open('./model/tokenizer.pkl', 'rb'))
model_score = load_model('./model/model_skor.keras')
    
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

my_data = {
    "low": {
        'Tidur': 9, 'Pola makan': 9, 'Meditasi': 10, 'Konseling': 9, 'Hotline': 8, 'Komunikasi': 9,
        'Identifikasi': 8, 'Atur emosi': 9, 'Hubungi orang tua': 8, 'Menggambar': 9, 'Olahraga': 10,
        'Sosialisasi': 9, 'Hobi': 9, 'Bantuan professional': 10, 'Perawatan diri': 9, 'Rekomendasi Umum': 7,
        'Hubungi teman': 8, 'Kelola waktu': 9, 'Atur keuangan': 7, 'Pengobatan': 10, 'Media sosial': 6
    },
    "moderate": {
        'Meditasi': 6, 'Hotline': 5, 'Sosialisasi': 5, 'Olahraga': 6, 'Bantuan professional': 6,
        'Rekomendasi Umum': 4, 'Hobi': 5, 'Komunikasi': 5, 'Konseling': 6, 'Tidur': 6, 'Perawatan diri': 6,
        'Afirmasi diri': 6, 'Pengobatan': 5, 'Identifikasi': 4, 'Media sosial': 3, 'Atur keuangan': 5,
        'Hubungi orang tua': 5, 'Hubungi teman': 5, 'Atur emosi': 4, 'Kelola waktu': 5, 'Kafein dan alkohol': 2,
        'Pola makan': 4
    },
    "high": {
        "Komunikasi": -2, "Konseling": -1, "Meditasi": -1, "Tidur": 0, "Bantuan professional": -2,
        "Rekomendasi Umum": -3, "Olahraga": -1, "Identifikasi": -4, "Media sosial": -8, "Sosialisasi": -3,
        "Perawatan diri": -2, "Hotline": -5, "Hubungi teman": -3, "Atur keuangan": -6, "Afirmasi diri": -2,
        "Pengobatan": -3, "Hubungi orang tua": -2, "Atur emosi": -4, "Hobi": -1, "Kelola waktu": -4,
        "Pola makan": -2, "Kafein dan alcohol": -10
    }
}

# Menyiapkan data untuk digunakan dalam model
def prepare_data(data):
    rows = []
    scores = []
    categories = list(data.keys())
    
    for category in categories:
        for activity, score in data[category].items():
            # Menambahkan data untuk kategori, aktivitas, dan skor
            rows.append([category, activity])
            scores.append(score)
    
    # Mengonversi data menjadi numpy array
    return np.array(rows), np.array(scores)

# Menyiapkan data untuk model
X, y = prepare_data(my_data)

# Menyusun fitur: Kategori dan Aktivitas
category_encoder = {category: idx for idx, category in enumerate(np.unique(X[:, 0]))}
activity_encoder = {activity: idx for idx, activity in enumerate(np.unique(X[:, 1]))}


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



def predict_score(category, activity):
    # Encode input kategori dan aktivitas
    encoded_input = np.array([[category_encoder[category], activity_encoder[activity]]])
    predicted_score = model_score.predict(encoded_input)
    
    number = max(-10, min(10, predicted_score[0][0]))
    
    return number


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
    get_scores = predict_score(predicted_stress.lower(), recommendations[0]['icon'])
    
    return {
        "predicted_stress": {
            "stress_level": predicted_stress,
            "score": f"{get_scores:.2f}"
        },
        "recommendations": recommendations
    }
