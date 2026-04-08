import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Cargar los datos que creamos
df = pd.read_csv('datos_transporte.csv')

# 2. Convertir letras a números para que la IA entienda
df_numeric = pd.get_dummies(df, columns=['Origen', 'Destino'])

# 3. Separar variables (X) del resultado (y)
X = df_numeric.drop('Retraso', axis=1)
y = df_numeric['Retraso']

# 4. Crear el Árbol (Usando Entropía como dice Palma Méndez)
modelo = DecisionTreeClassifier(criterion='entropy')
modelo.fit(X, y)

# 5. Dibujar el resultado
plt.figure(figsize=(12,8))
plot_tree(modelo, feature_names=X.columns, class_names=['A tiempo', 'Retrasado'], filled=True)
plt.title("Modelo de Predicción de Retrasos - Transporte Masivo")
plt.show()