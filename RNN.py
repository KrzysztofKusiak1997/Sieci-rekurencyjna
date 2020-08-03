# -*- coding: utf-8 -*-

### Model rekurencyjnej sieci neuronowej prognozujący liczbę potwierdzonych przypadków
### zachorowania a covid-29 w Polsce na podstawie przebiegów z krajów, które znajdują się
### w dalszym stadium epidemii. Wzięto pod uwagę kraje : Czechy, Austria oraz Szwajcaria.
### Dane pobrano ze strony: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset

#importowanie bibliotek
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#wczytanie danych i odrzucenie niepotrzebnych kolumn
dataset = pd.read_csv('covid_19_data.csv')
dataset = dataset.drop(dataset.columns[[0, 2, 4, 6, 7]], axis = 1)

#stworzenie podzbiorów danych z poszczególnych państw
dataset_train1 = dataset[dataset.Country_Region == "Czech Republic"]
dataset_train2 = dataset[dataset.Country_Region == "Austria"]
dataset_train3 = dataset[dataset.Country_Region == "Switzerland"]
dataset_test = dataset[dataset.Country_Region == "Poland"]

#odrzucenie niepotrzebnych kolumn
dataset_train1 = dataset_train1.drop(dataset.columns[[0, 1]], axis = 1)
dataset_train2 = dataset_train2.drop(dataset.columns[[0, 1]], axis = 1)
dataset_train3 = dataset_train3.drop(dataset.columns[[0, 1]], axis = 1)
dataset_test = dataset_test.drop(dataset.columns[[0, 1]], axis = 1)

#zresetowanie indeksów
dataset_train1 = dataset_train1.reset_index()
dataset_train2 = dataset_train2.reset_index()
dataset_train3 = dataset_train3.reset_index()
dataset_test = dataset_test.reset_index()

#odrzucenie kolumny z indeksami
dataset_train1 = dataset_train1.drop(dataset_train1.columns[[0]], axis = 1)
dataset_train2 = dataset_train2.drop(dataset_train2.columns[[0]], axis = 1)
dataset_train3 = dataset_train3.drop(dataset_train3.columns[[0]], axis = 1)
dataset_test = dataset_test.drop(dataset_test.columns[[0]], axis = 1)

#skalowanie danych
sc = MinMaxScaler(feature_range = (0, 1))
dataset_scaled1 = sc.fit_transform(dataset_train1)
dataset_scaled2 = sc.fit_transform(dataset_train2)
dataset_scaled3 = sc.fit_transform(dataset_train3)
dataset_test_scaled = sc.fit_transform(dataset_test)

#utworzenie list z wektorami wejsciowymi i wyjsciowymi
X_train1 = []
y_train1 = []

X_train2 = []
y_train2 = []

X_train3 = []
y_train3 = []

for i in range(7, 54):
    X_train1.append(dataset_scaled1[i-7:i])
    y_train1.append(dataset_scaled1[i])

for i in range(7, 59):
    X_train2.append(dataset_scaled2[i-7:i])
    y_train2.append(dataset_scaled2[i])
    X_train3.append(dataset_scaled3[i-7:i])
    y_train3.append(dataset_scaled3[i])


X_test = []
y_true = []

for i in range(7, 51):
    X_test.append(dataset_test_scaled[i-7:i])
    y_true.append(dataset_test_scaled[i])


#zamienienie list na numpy array
X_train1, y_train1 = np.array(X_train1), np.array(y_train1)
X_train2, y_train2 = np.array(X_train2), np.array(y_train2)
X_train3, y_train3 = np.array(X_train3), np.array(y_train3)
X_test, y_true = np.array(X_test), np.array(y_true)

#zmienie liczby wymiarów tensorów (LSTM przyjmuje 3-wymiarowe tensory na wejsciu)
X_train1 = np.reshape(X_train1, (X_train1.shape[0], X_train1.shape[1], 1))
X_train2 = np.reshape(X_train2, (X_train2.shape[0], X_train2.shape[1], 1))
X_train3 = np.reshape(X_train3, (X_train3.shape[0], X_train3.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#imporytowanie modułów do stworzenia modelu RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#tworzenie modelu RNN
regressor = Sequential()
regressor.add(LSTM(units = 500, return_sequences = True, input_shape = (X_train1.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 500, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 500, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 500))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))

#kompilowanie modelu oraz jego nauka na podstawie danych z Czech, Austrii oraz Szwajcarii
regressor.compile(optimizer = "RMSprop", loss = 'mean_squared_error')
regressor.fit(X_train1, y_train1, epochs = 100, batch_size = 4)
regressor.fit(X_train2, y_train2, epochs = 100, batch_size = 4)
regressor.fit(X_train3, y_train3, epochs = 100, batch_size = 4)

#zapisanie modelu, aby nie trzeba było ponownie go uczycć
regressor.save('RNN_model.h5')

#wczytanie modelu, jeżeli mamy już nauczony
from keras.models import load_model
regressor = load_model('RNN_model.h5')

#predycja, skalowanie wyników do poprzedniej skali rzeczywistej
predicted_value = regressor.predict(X_test)
predicted_value = sc.inverse_transform(predicted_value)
predicted_value = predicted_value[:37]
y_true = sc.inverse_transform(y_true)
y_true = y_true[7:]

#zapisanie wynikóW w DataFrame
results = np.concatenate([y_true, predicted_value], axis = 1)
results = pd.DataFrame(data = results, dtype = np.int16, columns = ['y_true', 'y_pred'])

#zapisanie wyników do arkusza kalkulacyjnego
results.to_excel('results.xlsx')



