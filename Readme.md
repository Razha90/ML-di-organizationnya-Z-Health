# Routing
- /predict

# Contoh Request Data
```
## Requested With Json Data
{
    "waktu_belajar": 5.5,
    "waktu_belajar_tambahan": 2,
    "waktu_tidur": 8,
    "aktivitas_sosial": 3,
    "aktivitas_fisik": 1,
    "jurnal_harian": "Saya belajar keras hari ini"
}
```

# Output Request If Sucess
```
{
  "data": {
    "predicted_stress": {
      "score": "2.34",
      "stress_level": "High"
    },
    "recommendations": [
      {
        "description": "Terapis dapat membantu Anda memproses emosi yang kompleks terkait dengan perpisahan dan mengembangkan mekanisme koping yang sehat.",
        "icon": "Konseling",
        "title": "Pertimbangkan Konseling"
      }
    ]
  },
  "status": "success"
}
```

# Output Request If Error
```
{
  "errors": {
    "waktu_belajar": "Waktu belajar harus berupa angka (int atau float)."
  },
  "status": "error"
}
```