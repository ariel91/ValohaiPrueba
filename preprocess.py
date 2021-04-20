# Regresión Lineal Simple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import valohai
import os 
import json
import warnings
warnings.filterwarnings("ignore")
# Importar el data set

#dataset = pd.read_csv('Salary_Data.csv')

#Get the path to the folder where Valohai inputs are
input_path = os.getenv('VH_INPUTS_DIR', '.inputs/')
# Get the file path of our MNIST dataset that we defined in our YAML
dataset_path = os.path.join(input_path, 'salary_yaml/Salary_Data.csv')
dataset = pd.read_csv(dataset_path)
#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 1].values
X = dataset.drop("Salary",axis=1)
y = dataset["Salary"]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

output_path = os.getenv('VH_OUTPUTS_DIR', '.outputs/')
OutDataset = os.path.join(output_path, 'carpeta_dataset_salida/train_dataframe.csv')

X_train["Salary"]=y_train.tolist()
train = X_train
#train.to_csv ("train_dataframe.csv", index = False, header=True)
train.to_csv (OutDataset, index = False, header=True)
print(OutDataset)