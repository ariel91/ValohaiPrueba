# Regresión Lineal Simple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import valohai
import os 
import json

# Importar el data set

#dataset = pd.read_csv('Salary_Data.csv')


#Get the path to the folder where Valohai inputs are
input_path = os.getenv('VH_INPUTS_DIR', '.inputs/')
# Get the file path of our MNIST dataset that we defined in our YAML
dataset_path = os.path.join(input_path, 'salary_yaml/Salary_Data.csv')
dataset = pd.read_csv(dataset_path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Crear modelo de Regresión Lienal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el conjunto de test
y_pred = regression.predict(X_test)
df = pd.DataFrame(y_pred)
df.to_csv (r'/valohai/outputs/export_dataframe.csv', index = False, header=True)
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