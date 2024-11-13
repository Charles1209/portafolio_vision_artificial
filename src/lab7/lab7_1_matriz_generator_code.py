from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Get predictions
y_train_pred = model.predict(hog_descriptors_train)[1].ravel()
y_test_pred = model.predict(hog_descriptors_test)[1].ravel()

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot training confusion matrix
ConfusionMatrixDisplay.from_predictions(
    labels_train, 
    y_train_pred,
    ax=ax1,
    cmap='Blues',
    normalize='true',
    display_labels=range(10)
)
ax1.set_title('Matriz de Confusión (Training)')

# Plot testing confusion matrix
ConfusionMatrixDisplay.from_predictions(
    labels_test, 
    y_test_pred,
    ax=ax2,
    cmap='Blues',
    normalize='true',
    display_labels=range(10)
)
ax2.set_title('Matriz de Confusión (Testing)')

plt.tight_layout()
plt.show()