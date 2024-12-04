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
    "predicted_stress": "Moderate",
    "recommendations": [
      {
        "description": "Mengurangi kontak akan melindungi Anda dari manipulasi dan stres lebih lanjut.",
        "icon": "Rekomendasi Umum",
        "title": "Batasi atau hentikan semua kontak dengan mantan Anda."
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