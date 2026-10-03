# Denses-Flow
​"A lightweight, zero-dependency JavaScript micro-framework for building and training Dense Neural Networks (MLP) from scratch."

# Denses Flow (DF) Library Documentation

Denses Flow (df) adalah sebuah library Machine Learning kustom berbasis JavaScript yang dirancang ringan, efisien, dan fleksibel untuk membangun, melatih, serta mengoptimalkan Jaringan Saraf Tiruan (Neural Network) berjenis Dense / Fully Connected (MLP).

---

## 📦 Instalasi & Penggunaan (CDN)

Kamu bisa mengimpor Denses Flow langsung melalui CDN favorit kamu:

<!-- Masukkan link CDN kamu di sini -->
<script src="https://your-cdn-link-here/df.js"></script>

Atau jika menggunakan modul ES6:

import { df } from "https://your-cdn-link-here/df.js";

Perhatian: Karena MD ini di generate oleh AI dan gk begitu jelas, gk ada link ke CDN atau apapun yang akan saya masukan ke readme, soalnya kan bisa di copas aja di df.js lagian kodenya cuma < 15kb kok, ya.

---

## 📑 Daftar Isi

1. Pemrosesan Dataset
2. Fungsi Aktivasi & Propagasi Balik (Forward & Backward)
3. Inisialisasi Bobot & Distribusi (Weight Initialization)
4. Arsitektur & Inisialisasi Model Denses
5. Inspeksi & Prediksi Model
6. Fungsi Loss / Cost Criterion
7. Regularisasi Weight Decay (L12)
8. Modul Optimizer & Management States
9. High-Level Training API

---

## 1. Pemrosesan Dataset

Fungsi df.objectData digunakan untuk mengonversi array object mentah yang berisi pasangan input-output menjadi struktur dataset matriks terpisah (x dan y).

### Pemanggilan:
const dataset = df.objectData(dataObject);

### Contoh Format:
const rawData = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] }
];

const dataset = df.objectData(rawData);
// Hasil Output: { x: [[0,0], [0,1]], y: [[0], [1]] }

---

## 2. Fungsi Aktivasi & Propagasi Balik (Forward & Backward)

Setiap fungsi aktivasi memiliki dua metode utama: .forward(input) dan .backward(error, outputForward).

* df.linear
  - Forward: Memetakan nilai gradien linear secara langsung.
  - Backward: Meneruskan error tanpa modifikasi non-linear.
* df.relu
  - Forward: Memotong nilai di bawah 0 menjadi 0 (f(x) = max(0, x)).
  - Backward: Mengalirkan gradien hanya jika x > 0.
* df.sigmoid
  - Forward: Memetakan nilai ke dalam rentang skala 0 hingga 1.
  - Backward: Menggunakan turunan f'(x) = f(x) * (1 - f(x)).
* df.tanh
  - Forward: Memetakan nilai ke dalam rentang skala -1 hingga 1.
  - Backward: Menggunakan turunan f'(x) = 1 - f(x)^2.
* df.softmax
  - Forward: Normalisasi eksponensial untuk distribusi probabilitas multi-kelas.
  - Backward: Menghitung gradien berbasis matriks distribusi softmax yang dihasilkan.

---

## 3. Inisialisasi Bobot & Distribusi (Weight Initialization)

Menyediakan mekanisme generator acak berbasis matematika formal untuk memicu stabilitas konvergensi saat pelatihan dimulai:

* df.normal(): Menghasilkan sampel acak berbasis Distribusi Normal Standar.
* df.HEN(size): He Normal Initialization untuk aktivasi berbasis ReLU.
* df.HEU(size): He Uniform Initialization.
* df.xaviern(fin, fout): Xavier/Glorot Normal Initialization untuk aktivasi Sigmoid/Tanh.
* df.xavieru(fin, fout): Xavier/Glorot Uniform Initialization.

---

## 4. Arsitektur & Inisialisasi Model Denses

Fungsi df.denses digunakan untuk memicu pembuatan objek model neural network. Dapat dipanggil dengan dua jenis penamaan properti objek konfigurasi:

### Opsi Gaya 1:
const model = df.denses({
    layers: [2, 4, 1],
    activations: [df.linear, df.relu, df.sigmoid],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

### Opsi Gaya 2 (Alias):
const model = df.denses({
    arsitect: [2, 4, 1],
    activates: [df.linear, df.relu, df.sigmoid],
    randoms: [null, df.HEN, df.xavieru]
});

Catatan Arsitektur: Struktur model yang dihasilkan memuat array flat model.params yang menampung seluruh bobot (weights) dan bias secara berurutan.

---

## 5. Inspeksi & Prediksi Model

* df.walk(model, callback)
  Menelusuri arsitektur internal model layer demi layer.
  Contoh: df.walk(model, info => console.log(info.i, info));

* df.predict(input, model)
  Melakukan forward pass ringkas dan hanya mengembalikan hasil output layer terakhir.

* df.forward(input, model)
  Melakukan forward pass lengkap dan mengembalikan array berisi semua matriks aktivasi di setiap layer (objek box).

* df.backward(error, box, model)
  Menerima error dari output layer, melacak balik lewat matriks box, dan mengembalikan:
  - gradients: Turunan gradien flat yang berkorespondensi langsung dengan model.params.
  - errorInput: Akumulasi error yang sampai kembali ke layer input pertama.

---

## 6. Fungsi Loss / Cost Criterion

Fungsi loss digunakan untuk mengevaluasi error akhir dari layer keluaran jaringan. Mengembalikan nilai .loss dan .error (gradien loss terhadap output).

* df.LIN(output, target): Evaluasi deviasi linear langsung (Internal Linear Bias Loss). Mengembalikan array loss terpisah [underpredict, overpredict].
* df.MSE(output, target): Mean Squared Error (Sangat baik untuk Regresi).
* df.MAE(output, target): Mean Absolute Error.
* df.BCE(output, target): Binary Cross-Entropy (Untuk Klasifikasi Biner 0 atau 1).
* df.CCE(output, target): Categorical Cross-Entropy (Untuk Klasifikasi Multi-kelas).

---

## 7. Regularisasi Weight Decay (L12)

Menerapkan penalti Regularisasi L1 (Lasso) dan L2 (Ridge) atau kombinasinya (Elastic Net) langsung pada array gradients sebelum diteruskan ke optimizer.

### Pemanggilan:
df.L12(gradients, model, config);

### Parameter Konfigurasi:
- config.lambda (Number, Default: 0.001): Kekuatan penalti/regularisasi (rate)
- config.ratio (Number, Default: 0.5): Rasio pembagi Elastic Net (0 = L2 Only, 1 = L1 Only)

---

## 8. Modul Optimizer & Management States

Denses Flow menggunakan pemisahan state yang efisien untuk menghindari kebocoran memori (memory leak). Setiap optimizer wajib dipasangkan dengan fungsi inisialisasi state-nya di awal pelatihan.

* Stochastic Gradient Descent (SGD)
  - State: const states = df.SGD_states(model, { lr: 0.01 })
  - Update: df.SGD(gradients, model, states)

* Momentum (MOM)
  - State: const states = df.MOM_states(model, { lr: 0.01, beta: 0.9 })
  - Update: df.MOM(gradients, model, states)

* RMSprop (RMSp)
  - State: const states = df.RMSp_states(model, { lr: 0.001, beta: 0.99, eps: 1e-8 })
  - Update: df.RMSp(gradients, model, states)

* Adam Optimizer (adam)
  - State: const states = df.adam_states(model, { lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8 })
  - Update: df.adam(gradients, model, states)

---

## 9. High-Level Training API

Untuk menyederhanakan training loop manual, Denses Flow menyediakan fungsi pembungkus tingkat tinggi:

* df.epoch(max_epoch, callback)
  Mengontrol jalannya iterasi training global. Jika callback mengembalikan nilai true, loop akan berhenti lebih awal (Early Stopping).

* df.Train(config)
  Mengotomatisasi seluruh alur kerja forward, backward, regularisasi, kalkulasi loss, hingga langkah optimasi.

### Contoh Implementasi Final Terpadu (XOR Gates):

const xorData = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] }
];

const modelFinal = df.denses({
    layers: [2, 8, 1],
    activations: [df.linear, df.relu, df.sigmoid]
});

const datasetFinal = df.objectData(xorData);
const statesFinal = df.adam_states(modelFinal);

df.epoch(5000, epc => {
    let lossNow;

    df.Train({
        dataset: datasetFinal,
        model: modelFinal,
        states: statesFinal,
        loss: df.MSE,
        gradients: (grads, model, state) => {
            // Menggunakan L12 dengan default { lambda: 0.001, ratio: 0.5 }
            df.L12(grads, model, { lambda: 0.001, ratio: 0.5 }); 
            df.adam(grads, model, state);
        },
        totalloss: loss => {
            lossNow = loss;
        }
    });

    if (lossNow < 0.01) return true; // Early Stopping
});
