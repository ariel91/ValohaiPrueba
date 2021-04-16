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

outputs_path = os.getenv('VH_OUTPUTS_DIR', './outputs')
output_path = os.path.join(outputs_path, 'preprocessed_salary.csv')
df.to_csv(output_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
#df.to_csv (r'/valohai/outputs/export_dataframe.csv', index = False, header=True)