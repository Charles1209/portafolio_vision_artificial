from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import load_digits
from sklearn.svm import SVC

# Cargar dataset
digits = load_digits()

# Preparar datos
X, y = digits.data, digits.target

# K-Fold con k=25
kfold = KFold(n_splits=25, random_state=11, shuffle=True)

# Modelo SVM con RBF
svclassifier = SVC(kernel='rbf', gamma='auto')

# Cross Validation
scores = cross_val_score(estimator=svclassifier, X=X, y=y, cv=kfold)

print(f'Exactitud promedio (k=25): {scores.mean():.2%}')
print(f'Desviación estándar: {scores.std():.2%}')