# Importaciones necesarias
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Cargar dataset
digits = load_digits()

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11)

# Definir clasificador SVM con kernel RBF
svclassifier = SVC(kernel='rbf', gamma='auto')

# Entrenar modelo
svclassifier.fit(X_train, y_train)

# Predicciones
predicted = svclassifier.predict(X_test)
expected = y_test

# Precisión del modelo
precision = svclassifier.score(X_test, y_test)
print(f'Precisión con kernel RBF: {precision:.2%}')

# Reporte de clasificación
names = [str(digit) for digit in digits.target_names]
print(classification_report(expected, predicted, target_names=names))