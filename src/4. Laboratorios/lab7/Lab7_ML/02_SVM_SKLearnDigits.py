# Clasificación utilizando SVM y el dataset digits

########################################
####	Descripción del Dataset		####
########################################

#%matplotlib inline
from sklearn.datasets import load_digits

digits = load_digits()

print(digits.DESCR)

####	Verificación del tamaño de las secuencias
digits.target[::100]

digits.data.shape

digits.target.shape

####	Una muestra del Dataset Digits
digits.images[13]

####	Preparación de los datos para utilizarlos con SciKit-Learn
digits.data[13]

########################################
#### Visualización de los Datos		####
########################################

####	Creación del diagrama
import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))

### Displaying Each Image and Removing the Axes Labels 

for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])  # remove x-axis tick marks
    axes.set_yticks([])  # remove y-axis tick marks
    axes.set_title(target)
plt.tight_layout()     

####################################################################
####	Dividiendo en dataset en secuencias de Train y Test		####
####################################################################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11)

####	Tamaño de las secuencias de Train y Test
X_train.shape

X_test.shape

####################################
####	Creación del Modelo		####
####################################

#@title Celda 1

########################################
####	Entrenamiento del Modelo	####
########################################

# Definición del estimador
from sklearn.svm import SVC

svclassifier = SVC(kernel='poly', degree=8, gamma='auto')

svclassifier.fit(X=X_train, y=y_train)

################################################################
####	Haciendo predicciones para las clases de Digits		####
################################################################

predicted = svclassifier.predict(X=X_test)

expected = y_test

predicted[:20]

expected[:20]

# Listado de predicciones equivocadas
wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]

wrong

# Precisión lograda por el clasificador
print(f'{svclassifier.score(X_test, y_test):.2%}')

####	Matriz de Confución

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)

confusion

####	Reporte de Clasificación
from sklearn.metrics import classification_report

names = [str(digit) for digit in digits.target_names]

print(classification_report(expected, predicted, target_names=names))

####	Visualizar la Matriz de Confución
import pandas as pd

confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))

import seaborn as sns

axes = sns.heatmap(confusion_df, annot=True, cmap='nipy_spectral_r')

################################
####	Validación Cruzada	####
################################

from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=11, shuffle=True)

####	Usar el objeto KFold con la función cross_val_score
#@title Celda 2

from sklearn.model_selection import cross_val_score

# Aplicar Cross Validation con la función cross_val_score
scores = cross_val_score(estimator=svclassifier, X=digits.data, y=digits.target, cv=kfold)

scores

print(f'Mean accuracy: {scores.mean():.2%}')

print(f'Accuracy standard deviation: {scores.std():.2%}')

################################################################
####	Ejecutando varios modelos para encontrar el mejor	####
################################################################

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

# Instancias de Clasificadores a evaluar
estimators = {
     'KNeighborsClassifier': KNeighborsClassifier(), 
     'SVC': svclassifier,
     'GaussianNB': GaussianNB()}

for estimator_name, estimator_object in estimators.items():
     kfold = KFold(n_splits=10, random_state=11, shuffle=True)
     scores = cross_val_score(estimator=estimator_object, 
         X=digits.data, y=digits.target, cv=kfold)
     print(f'{estimator_name:>20}: ' + 
           f'mean accuracy={scores.mean():.2%}; ' +
           f'standard deviation={scores.std():.2%}')