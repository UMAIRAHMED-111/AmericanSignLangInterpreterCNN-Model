import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Set your data directory
data_dir = './dataset'

# Hyperparameters
batch_size = 32
input_shape = (64, 64, 3)

# Load the saved model
model = load_model('asl_cnn_model.h5')

# Prepare the data generator for evaluation
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)  # Important: set shuffle to False for proper evaluation

# Evaluate the model on the test set
scores = model.evaluate(test_generator)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Make predictions on the test set
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels from the test set
y_true = test_generator.classes

# Calculate classification metrics
class_labels = list(test_generator.class_indices.keys())
report = classification_report(y_true, y_pred_classes, target_names=class_labels)
conf_mat = confusion_matrix(y_true, y_pred_classes)

print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_mat)

