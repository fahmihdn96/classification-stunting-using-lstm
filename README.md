**Judul:** Klasifikasi Stunting Menggunakan LSTM

**Deskripsi:**

Repository ini berisi kode untuk klasifikasi stunting menggunakan LSTM. Data yang digunakan adalah data antropometri anak yang didapatkan dari Dinkes Kota Bogor. Model dilatih menggunakan dataset yang dibagi menjadi 80% untuk pelatihan dan 20% untuk pengujian. Anda bisa melakukan beberapa kali pengujian pada hyperparameter untuk menemukan nilai akurasi terbaik.

Model dideploy ke dalam Streamlit sehingga dapat digunakan secara online.

**Streamlit:**

https://classification-stunting-using-lstm-mw9lytbvdz3tbqqqxwxiwi.streamlit.app/

**Daftar file:**

* `app.py`: File utama aplikasi Streamlit.
* `Stunting.py`: File yang berisi kode model LSTM.
* `stunting.csv`: Data yang digunakan untuk melatih model.

**Daftar dependensi:**

* Python 3.8 atau lebih tinggi
* Streamlit
* NumPy
* Pandas
* Seaborn
* Matplotlib
* Scikit-learn
* TensorFlow
