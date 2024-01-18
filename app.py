import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization
from tensorflow.keras.layers import Dropout

# Buat tampilan web menggunakan Streamlit
st.title('Aplikasi Klasifikasi Stunting')

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Tampilkan dataset yang diunggah
    st.header('Dataset Stunting Dinkes Kota Bogor')
    st.write(data)
    
    # Ganti nilai 0 pada kolom bb lahir dan tb lahir menjadi missing value
    data['BB_Lahir'].replace(0, np.nan, inplace=True)
    data['TB_Lahir'].replace(0, np.nan, inplace=True)
    # Hapus data kolom kosong
    data = data.dropna()
    # Hapus kolom 'BB_U', 'ZS_BB_U', 'BB_TB', 'ZS_BB_TB'
    data = data.drop(['BB_Lahir', 'TB_Lahir', 'BB_U', 'ZS_BB_U', 'BB_TB', 'ZS_BB_TB'], axis=1)
    
    #LabelEncoding
    encode = LabelEncoder()
    data['JK'] = encode.fit_transform(data['JK'].values)
    data['TB_U'] = encode.fit_transform(data['TB_U'].values)
    data['Status'] = encode.fit_transform(data['Status'].values)

    # Pisahkan fitur dan target
    st.header('Data Selection')
    X = data[['JK', 'Umur', 'Berat', 'Tinggi', 'ZS_TB_U']]
    st.write('Features (X):')
    st.write(X)
    y = data['Status']
    st.write('Target (y):')
    st.write(y)
    # Gabungkan X dan y
    # merged_data = pd.concat([X, y], axis=1)
    # st.write('Merged Data:')
    # st.write(merged_data)

    # Bagi data menjadi data pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Tampilkan ukuran data pelatihan dan data uji
    st.write(f'Size of Training Data: {X_train.shape[0]} samples')
    st.write(f'Size of Testing Data: {X_test.shape[0]} samples')
    
    #handling Imbalace Data
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Normalisasi data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Menambahkan dimensi waktu ke data train dan test
    X_train = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    
    # Tambahkan input parameter
    st.header('Pilih Parameter Training')
    available_neurons = [16, 32, 64, 128, 256, 512, 1024]
    available_epochs = [10, 20, 50, 100, 200, 500]
    available_batch_sizes = [32, 64, 128, 256, 512, 1024]
    neurons = st.select_slider('Jumlah Neuron', options=available_neurons)
    epochs = st.select_slider('Epoch', options=available_epochs)
    batch_size = st.select_slider('Batch Size', options=available_batch_sizes)

    # Tambahkan tombol untuk melatih model
    if st.button('Latih Model'):
        # Buat model LSTM
        model = Sequential()
        model.add(LSTM(neurons, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(neurons, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        
        # Training model dengan menambahkan validation split untuk memonitor val loss
        history = model.fit(X_train,
                    y_train_resampled,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test))

        # Evaluasi model
        loss, accuracy = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_binary)
        
        st.write(f'Akurasi Model: {accuracy:.2f}')

        # Visualisasi Train Loss dan Validation Loss
        st.header('Visualisasi Loss')
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

        # Perbandingan Y Test dan Y Prediksi
        st.header('Perbandingan Data Test dan Hasil Model LSTM')
        comparison_df = pd.DataFrame({'Data Test': y_test, 'Data Hasil Model LSTM': y_pred_binary.flatten()})
        st.write(comparison_df)

        # Confusion Matrix
        st.header('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred_binary)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('LSTM')
        ax.set_ylabel('Aktual')
        st.pyplot(fig)

        # Metrik Lainnya
        st.header('Classification Report')
        classification_rep = classification_report(y_test, y_pred_binary)
        st.text(classification_rep)