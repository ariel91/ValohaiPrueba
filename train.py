# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import valohai
import os 
import json
import warnings
warnings.filterwarnings("ignore")

#OBTENCIÓN DE TRAIN Y TEST DESDE VALOHAI A TRAVÉS DE ARCHIVO YAML

#Get the path to the folder where Valohai inputs are
input_path = os.getenv('VH_INPUTS_DIR', '.inputs/')
# Get the file path of our MNIST dataset that we defined in our YAML
dataset_train_path = os.path.join(input_path, 'train_yaml/train_dataframe.csv')

#Get the path to the folder where Valohai inputs are
input_path = os.getenv('VH_INPUTS_DIR', '.inputs/')
# Get the file path of our MNIST dataset that we defined in our YAML
dataset_test_path = os.path.join(input_path, 'test_yaml/test_dataframe.csv')


dataset_train = pd.read_csv(dataset_train_path)
#dataset_train= dataset_train.rename(columns={'YearsExperience':'X_train',
#                                   'Salary':'y_train'})
X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, 1].values

dataset_test = pd.read_csv(dataset_train_path)
#dataset_train= dataset_train.rename(columns={'YearsExperience':'X_train',
#                                   'Salary':'y_train'})
X_test = dataset_test.iloc[:, :-1].values
y_test = dataset_test.iloc[:, 1].values

# Crear modelo de Regresión Lienal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
# Predecir el conjunto de test
y_pred = regression.predict(X_test)
df = pd.DataFrame(y_pred)
df.to_csv (r'/valohai/outputs/predict_dataframe.csv', index = False, header=True)
print(y_pred)

# Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")

save_path1 = valohai.outputs().path('myplot1.png')
plt.savefig(save_path1)

plt.show()
plt.close()

# Visualizar los resultados de test
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")


save_path2 = valohai.outputs().path('myplot2.png')
plt.savefig(save_path2)

plt.show()
plt.close()