# WZUM_zaliczenie
Sign language letter recognition project


Celem projektu było ztworzenie datasetu liter w języku migowym za pomocą specjalnego programu, przetworzenie danych oraz wytrenowanie modelu klasyfikatora. 
Dla poprawy dokładności klasyfikatora zakodowana została kolumna opisująca, która ręka została użyta do zrobienia litery. Normalizacja danych została wykonanna za pomocą funkcji MinMaxScaler oraz StandardScaler. Zbiór został podzielony na treningowy oraz testowy. 
Do klasyfikacji wykorzystano klasyfikator z wieloma warstwami neutronowymi (MLPClassifier).
Jest to model, który optymalizuje funkcję straty przy użyciu algorytmu LBFGS. Jako funkcję aktywacji przyjęto funkcję sigmoidalną o maksymalnej liczbie iteracji 30000. 
Przeanalizowano także model w poszukiwaniu hiperparametrów oraz po odrzuceniu wartości odstających outliers. Funkcja GridSearch oraz HalvingGridSearch nie poprawiają skuteczności klasyfikacji. 
Do trenowaniu użyto także kalibracji prawdopodobieństw  przy użyciu regresji funkcją sigmoidalną.


Na zbiorze testowym klasyfikator uzyskał 95,5% skuteczności, zmierzone metryką f1_score.
