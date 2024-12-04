from model import getAllRecommended
from flask import Flask
from flask import request, jsonify



# lawak = getAllRecommended(7.5, 2.0, 2.0, 3.0, 1.5, "Saya merasa kelelahan karena tugas yang menumpuk.")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def hello_world():
  try:
    # Ambil data JSON
    data = request.get_json()

    # Validasi apakah data JSON ada
    if not data:
        return jsonify({"status": "error", "message": "Missing JSON payload"}), 400

    # Ambil field dari JSON
    waktu_belajar = data.get('waktu_belajar')
    waktu_belajar_tambahan = data.get('waktu_belajar_tambahan')
    waktu_tidur = data.get('waktu_tidur')
    aktivitas_sosial = data.get('aktivitas_sosial')
    aktivitas_fisik = data.get('aktivitas_fisik')
    jurnal_harian = data.get('jurnal_harian')

    # Validasi tipe data dan nilai
    errors = {}
    if not isinstance(waktu_belajar, (int, float)):
        errors['waktu_belajar'] = "Waktu belajar harus berupa angka (int atau float)."
    if not isinstance(waktu_belajar_tambahan, (int, float)):
        errors['waktu_belajar_tambahan'] = "Waktu belajar tambahan harus berupa angka (int atau float)."
    if not isinstance(waktu_tidur, (int, float)):
        errors['waktu_tidur'] = "Waktu tidur harus berupa angka (int atau float)."
    if not isinstance(aktivitas_sosial, (int, float)):
        errors['aktivitas_sosial'] = "Aktivitas sosial harus berupa angka (int atau float)."
    if not isinstance(aktivitas_fisik, (int, float)):
        errors['aktivitas_fisik'] = "Aktivitas fisik harus berupa angka (int atau float)."
    if not isinstance(jurnal_harian, str):
        errors['jurnal_harian'] = "Jurnal harian harus berupa string."

    # Jika ada error, kembalikan respon error
    if errors:
        return jsonify({"status": "error", "errors": errors}), 400

    # Jika validasi berhasil, panggil fungsi `getAllRecommended`
    result = getAllRecommended(waktu_belajar, waktu_belajar_tambahan, waktu_tidur, aktivitas_sosial, aktivitas_fisik, jurnal_harian)
    return jsonify({"status": "success", "data": result})

  except Exception as e:
      return jsonify({"status": "error", "message": str(e)}), 500