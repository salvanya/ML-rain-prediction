# Librerías generales
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Statsmodels y SciPy
from scipy.stats import pearsonr

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, r2_score,accuracy_score, precision_score, recall_score, f1_score)

# Imbalanced-learn
from imblearn.over_sampling import SMOTE

# Tensorflow
from tensorflow import keras

# Scikit-learn y Tensorflow (repetidos, puedes eliminar uno de cada par)
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor

# Funciones generales
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

# Guardar y cargar modelos
import joblib

"""# Importar y procesar datos para el entrenamiento y prueba de los modelos

## Importar el dataset
"""

weatherAUS_df = pd.read_csv('weatherAUS.csv')

"""## Procesar los datos

"""

# Función de procesamiento de datos
def process_data(df):

    # Eliminar columnas innecesarias ----------------------------------------------------------------------------------------------------------------
    # Eliminar la columna 'Unnamed: 0'
    df = df.drop(columns=['Unnamed: 0'])

    # Eliminar registros con datos nulos en las variables respuesta 'RainTomorrow' y 'RainfallTomorrow'
    df.dropna(subset=['RainTomorrow', 'RainfallTomorrow'], inplace=True)

    # Lista de ubicaciones de interés
    ubicaciones_interes = ["Sydney", "SydneyAirport", "Canberra", "Melbourne", "MelbourneAirport"]

    # Filtrar las filas en base a las ubicaciones interes
    df = df[df['Location'].isin(ubicaciones_interes)]

    # Eliminar direcciones del viento
    columns_to_exclude = [col for col in df.columns if "Dir" in col]
    df.drop(columns=columns_to_exclude, inplace = True)

    # Utilizar replace para cambiar el valor 9 por NaN en 'Cloud9am' --------------------------------------------------------------------------------
    df['Cloud9am'] = df['Cloud9am'].replace(9, np.nan)

    # Codificar la fecha y otras variables categóricas ---------------------------------------------------------------------------------------------
    # Crear una variable wet_month para saber si la fecha corresponde a un mes lluvioso o no
    df['Date'] = pd.to_datetime(df['Date'])
    df['wet_month'] = df['Date'].apply(lambda x: 1 if 5 <= x.month <= 10 else 0)
    df.drop(columns=['Date'], inplace=True)

    # Codificar las variables categóricas de si llovió en la fecha y la localización
    columns_to_encode = ['Location', 'RainToday', 'RainTomorrow']
    df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

    return df

x_train_processed = process_data(weatherAUS_df)

"""## Separar conjuntos de entrenamiento y de prueba"""

def split_datasets(df):
    # Definir las columnas explicativas (X) y las variables de respuesta (y) para regresión y clasificación
    features = df.drop(columns=['RainfallTomorrow', 'RainTomorrow_Yes'])
    target_reg = df['RainfallTomorrow']
    target_class = df['RainTomorrow_Yes']

    # Separar el conjunto de entrenamiento y prueba
    x_train, x_test, y_train_reg, y_test_reg, y_train_class, y_test_class = train_test_split(
        features, target_reg, target_class, test_size=0.2, random_state=7
    )

    return x_train, x_test, y_train_reg, y_test_reg, y_train_class, y_test_class

# Uso de la función con tu DataFrame
x_train, x_test, y_train_reg, y_test_reg, y_train_class, y_test_class = split_datasets(x_train_processed)

"""# Procesamiento de los datos"""

class DataProcessor(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.scaler = StandardScaler()
        self.knn_imputer = KNNImputer(n_neighbors=5)

    def impute_RainToday(self, df):
        # Imputar datos de RainToday
        df['RainToday_Yes'] = df.apply(lambda row: 1 if row['Rainfall'] >= 1.2 else 0 if pd.isna(row['RainToday_Yes']) else row['RainToday_Yes'], axis=1)
        return df

    def fit(self, X, y=None):
        # Obtener columnas númericas
        columnas_numericas = list(X.select_dtypes(include=['float64']).columns)

        # Ajustar el imputador K-NN
        self.knn_imputer.fit(X[columnas_numericas])

        # Ajustar el escalador
        self.scaler.fit(X[columnas_numericas])

        return self

    def transform(self, X):
        # Obtener columnas númericas
        columnas_numericas = list(X.select_dtypes(include=['float64']).columns)

        # Aplica el imputador a las columnas seleccionadas
        X[columnas_numericas] = self.knn_imputer.transform(X[columnas_numericas])

        # Llamar a la función impute_RainToday
        X = self.impute_RainToday(X)

        # Escalar características
        X[columnas_numericas] = self.scaler.transform(X[columnas_numericas])

        return X

"""# Modelos

## Regresión
"""

class NeuralNetworkPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_layers = 1
        self.num_neurons = 32
        self.epochs = 70
        self.batch_size = 32
        self.model = None
        self.scaler = None
        self.knn_imputer = None
        self.best_params = None

    def fit(self, X, y):
        # Construir y compilar el modelo de la red neuronal
        self.model = self.build_model(X, self.num_layers, self.num_neurons)

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        # Realizar predicciones
        predictions = self.model.predict(X)
        return predictions

    def score(self, X, y, metric='r2'):
        # Calcular la métrica especificada (predicciones vs. valores reales)
        predictions = self.predict(X)

        if metric == 'r2':
            score = r2_score(y, predictions)
        elif metric == 'rmse':
            mse = mean_squared_error(y, predictions)
            score = np.sqrt(mse)
        else:
            raise ValueError("Métrica no válida. Use 'r2' o 'rmse'.")
        return score

    def build_model(self, X, num_layers, num_neurons):
        model = keras.Sequential()
        model.add(keras.layers.Dense(num_neurons, activation='relu', input_shape=(X.shape[1],)))

        for _ in range(num_layers - 1):
            model.add(keras.layers.Dense(num_neurons, activation='relu'))

        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

"""## Clasificación"""

class ClassificationPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.smote_params = {'sampling_strategy': 'auto', 'random_state': 42}
        self.lr_params = {'solver': 'liblinear', 'max_iter': 200, 'class_weight': 'balanced', 'C': 0.007}
        self.smote = None
        self.lr_model = None
        self.y_test_pred = None

    def fit(self, X, y):
        # Aplicar SMOTE al conjunto de entrenamiento
        if self.smote_params:
            self.smote = SMOTE(**self.smote_params)
            X_resampled, y_resampled = self.smote.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y

        # Entrenar el modelo de regresión logística
        if self.lr_params:
            self.lr_model = LogisticRegression(**self.lr_params)
            self.lr_model.fit(X_resampled, y_resampled)
        else:
            raise ValueError("Se requieren parámetros para el modelo de regresión logística.")

        return self

    def transform(self, X):
        # No es necesario realizar transformaciones específicas
        return X

    def predict(self, X):
        # Predecir utilizando el modelo de regresión logística entrenado
        if self.lr_model:
            self.y_test_pred = self.lr_model.predict(X)
            return self.y_test_pred
        else:
            raise ValueError("El modelo de regresión logística no ha sido entrenado.")

    def calculate_metrics(self, y_true=None):
        if y_true is None:
            raise ValueError("Se requiere el conjunto de etiquetas verdaderas (y_true) para calcular las métricas.")

        if self.y_test_pred is None:
            raise ValueError("Primero debes realizar predicciones en los datos de prueba.")

        accuracy = accuracy_score(y_true, self.y_test_pred)
        precision = precision_score(y_true, self.y_test_pred)
        recall = recall_score(y_true, self.y_test_pred)
        f1 = f1_score(y_true, self.y_test_pred)

        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        return metrics



