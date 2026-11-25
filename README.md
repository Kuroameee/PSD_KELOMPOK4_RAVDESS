# Speech Emotion Recognition pada RAVDESS — Replikasi, Metode Pembanding, dan Analisis

Proyek 4 minggu mata kuliah **Pengolahan Sinyal Digital (PSD)**.
Repositori ini berisi notebook Minggu 1–3: replikasi baseline dari paper, metode pembanding, serta analisis hasil dan uji signifikansi.

---

## Ringkasan

Tugasnya adalah mereplikasi satu paper jurnal yang kodenya tersedia, lalu mengusulkan dan menguji metode pembanding pada protokol evaluasi yang sama.

- **Paper acuan (baseline):** Luna-Jiménez, C., dkk. (2022). *A Proposal for Multimodal Emotion Recognition Using Aural Transformers and Action Units on RAVDESS Dataset.* Applied Sciences, 12(1), 327.
- **Kode penulis paper:** [`MMEmotionRecognition`](https://github.com/cristinalunaj/MMEmotionRecognition) (lisensi MIT)
- **Dataset:** RAVDESS (Livingstone & Russo, 2018, PLOS ONE), 1.440 rekaman *speech* dari 24 aktor, 8 emosi
- **Protokol evaluasi:** *subject-wise 5-fold cross-validation* dengan pembagian aktor tetap seperti paper (seed 2020)

Baseline paper memakai *late fusion* dari tiga cabang: fine-tuned Wav2Vec2 (audio), bi-LSTM atas Action Units (visual), dan MLP atas rata-rata Action Units (visual).

---

## Isi Repositori

| Berkas | Minggu | Isi |
|---|---|---|
| `WEEK_1_replikasi.ipynb` | 1 | Replikasi baseline: model statis pada Action Units dan embedding audio, serta late fusion (target paper 86,70 %) |
| `week2_Metode_Pembanding.ipynb` | 2 | Metode pembanding E1–E5 (termasuk E3b) beserta rancangan eksperimen dan kriteria evaluasi |
| `Week2_Metode_Pembanding_E5.ipynb` | 2 | Run tambahan E5 dengan seed berbeda, untuk ensemble di Minggu 3 |
| `Week3_Hasil_Analisis.ipynb` | 3 | Tabel lintas metode dan fold, uji berpasangan (dengan koreksi Holm), confusion matrix, boxplot per kelas, kurva validasi, ensemble, dan keterbatasan |

> Minggu 4 (draft artikel) dikerjakan terpisah dan tidak termasuk di repositori ini.

---

## Metode yang Dijalankan

### Minggu 1 — Replikasi baseline

| Blok | Isi | Rujukan di paper |
|---|---|---|
| A | Model statis (SVC, k-NN, MLP) pada rata-rata Action Units | Tabel 2 |
| B | Model statis (SVC, k-NN, MLP) pada rata-rata embedding Wav2Vec2 512-d | Tabel 1 |
| C | Late fusion (Logistic Regression) atas posterior tiga cabang | Tabel 3, Gambar 7–8 |
| D | Fine-tuning dari data mentah (**tidak dijalankan**, perlu GPU dan data mentah) | Tabel 1/2 |

### Minggu 2 — Metode pembanding

| ID | Sumbu perbedaan | Perubahan dari baseline |
|---|---|---|
| E1 | Pengklasifikasi | Cabang visual: MLP80 diganti LogReg, SVC, Random Forest, ExtraTrees, HistGB |
| E2 | Arsitektur fusi | Kepala fusi alternatif, ablasi cabang, dan cabang AU dengan posterior *out-of-fold* |
| E3 | Fitur | Audio: MFCC + Δ + ΔΔ (240-d) dengan SVM/RF, tanpa transformer |
| E3b | Fitur lebih kaya + tuning | ±379 fitur audio dengan SVM, hiperparameter dipilih lewat CV-dalam per aktor |
| E4 | Arsitektur + optimasi | CNN kecil pada log-mel + SpecAugment (kanal *song*, tidak sebanding dengan paper) |
| E5 | Optimasi | Fine-tuning Wav2Vec2 dengan **checkpoint yang sama** seperti paper, resep optimasi berbeda, lalu fusi dengan posterior visual paper |

### Minggu 3 — Hasil dan analisis

Perbandingan lintas metode, paired t-test antar fold (dengan CI 95 % selisih, Cohen's dz, dan koreksi Holm), confusion matrix, boxplot recall per kelas, kurva validasi E5, variabilitas antar-run, dan ensemble E5.

---

## Hasil Utama

Akurasi rata-rata 5-fold CV (%). Semua metode memakai 1.440 rekaman *speech*, kecuali E4.

| Metode | Akurasi | Pembanding di paper |
|---|---|---|
| Replikasi fusi baseline | **86,70** | 86,70 |
| Audio FT-Wav2Vec2 (posterior paper) | 81,82 | 81,82 |
| Visual bi-LSTM (posterior paper) | 62,13 | 62,13 |
| E5 audio (R4, epoch terakhir) | 82,10 | 81,82 (audio paper) |
| E5 fusi (audio E5 + visual paper) | 85,57 | 86,70 (fusi paper) |
| E5 audio ensemble (5 run) | 82,33 | 81,82 (audio paper) |
| E5 fusi ensemble (5 run) | 86,73 | 86,70 (fusi paper) |
| E3 MFCC + SVC RBF | 51,07 | 53,12 (audio FE paper) |
| E3b fitur kaya + SVC | 53,05 | 53,12 (audio FE paper) |
| E4 CNN log-mel (song) | 82,02 | tidak sebanding |

**Catatan membaca hasil**

- Dengan n = 5 fold, daya uji statistik rendah. Selisih kecil (kurang dari ±1 poin) berada di dalam CI 95 % dan **bukan bukti perbaikan**.
- Variabilitas antar-run E5 dengan resep yang sama sekitar 1 poin (audio 81,45–82,38 %, fusi 85,57–86,18 %).
- E4 memakai kanal *song* (1.012 rekaman, 23 aktor, tanpa sebagian emosi), jadi angkanya tidak boleh dibandingkan langsung dengan paper.
- Semua angka dilaporkan pada **epoch terakhir**, bukan epoch terbaik pada fold uji.

---

## Cara Menjalankan

### Dependensi

- Python 3.10+
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`
- `torch`, `transformers`, `librosa` (untuk E3, E3b, E4, E5)

### Data

1. **RAVDESS** dari Kaggle: [`corry371/database-speech-emotion-recognition-with-ravdess`](https://www.kaggle.com/datasets/corry371/database-speech-emotion-recognition-with-ravdess) (atau dari Zenodo, record 1188976).
2. **Repo penulis paper** (`MMEmotionRecognition`) berisi fitur dan posterior yang dibundel penulis. Notebook akan melakukan `git clone` otomatis jika repo belum ada di dataset (butuh Internet).

### Menjalankan di Kaggle (disarankan untuk Minggu 2 dan 3)

1. Buat *New Notebook*, lalu **Import Notebook** (unggah berkas `.ipynb`).
2. **Add Data**: dataset RAVDESS di atas.
3. **Accelerator: GPU T4** dan **Internet: On** (wajib untuk E5; E1–E3 tidak butuh GPU).
4. **Run All**. Hasil tersimpan di `/kaggle/working/results`.

Notebook Minggu 2 dan 3 menulis sendiri modul pendukungnya lewat `%%writefile` (`rav_utils.py`, `kaggle_data.py`, `w2v_finetune.py`, dan lainnya), jadi tidak ada kode lain yang perlu diunggah.

### Menjalankan Minggu 1 (lokal)

1. Ekstrak `MMEmotionRecognition-main.zip` ke folder di samping notebook.
2. Pastikan `rav_utils.py` ada di folder yang sama dengan notebook (modul ini ditulis oleh notebook Minggu 2, lihat sel `%%writefile rav_utils.py`).
3. Ubah variabel `REPO_DIR` di sel Setup agar menunjuk ke folder repo di komputermu.
4. Run All. Cukup CPU, dan tidak perlu mengunduh RAVDESS.

### Urutan pengerjaan

```
WEEK_1_replikasi  →  week2_Metode_Pembanding  →  Week2_Metode_Pembanding_E5  →  Week3_Hasil_Analisis
```

`Week2_Metode_Pembanding_E5` (≈ 1,6 jam per seed di GPU T4) menghasilkan `ENS_runs.zip`. Berkas itu opsional dan hanya dipakai di bagian ensemble (1b) pada notebook Minggu 3; tanpanya bagian tersebut dilewati.

---

## Catatan Jujur dan Keterbatasan

- **Minggu 1:** fitur AU rata-rata, embedding audio, dan posterior untuk fusi memakai berkas rilisan penulis. Jadi angka 86,70 % memverifikasi tahap fusi dan evaluasi, bukan fine-tuning dari awal.
- **Angka paper optimistis:** di kode penulis, checkpoint audio dan early-stopping bi-LSTM dipilih memakai fold uji. Cabang visual pada fusi E5 memakai posterior paper tersebut, sehingga terjadi asimetri seleksi terhadap audio E5.
- **Bias fusi:** kepala fusi dilatih pada posterior data latih *in-sample*, seperti di kode penulis.
- **Bias seleksi:** kandidat dipilih dari CV yang sama dengan evaluasinya.
- **Variabilitas GPU:** hasil fine-tuning tidak deterministik antar-run.
- **E3/E3b/E4:** hanya akurasi per fold (dibulatkan 0,1) yang dibawa ke Minggu 3, sehingga F1 per fold dan confusion matrix-nya tidak dihitung ulang.
- Normalisasi per-aktor sengaja **tidak** dipakai karena bersifat transduktif dan tidak dipakai paper.

---

## Referensi

1. Luna-Jiménez, C., Kleinlein, R., Griol, D., Callejas, Z., Montero, J. M., & Fernández-Martínez, F. (2022). A Proposal for Multimodal Emotion Recognition Using Aural Transformers and Action Units on RAVDESS Dataset. *Applied Sciences*, 12(1), 327.
2. Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLOS ONE*, 13(5), e0196391.
3. Repositori penulis: https://github.com/cristinalunaj/MMEmotionRecognition (MIT License)

## Lisensi

Kode dalam repositori ini untuk keperluan akademik (tugas kuliah). Dataset RAVDESS dan kode `MMEmotionRecognition` mengikuti lisensi aslinya masing-masing.
