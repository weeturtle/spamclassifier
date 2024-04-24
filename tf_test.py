import tensorflow as tf
from keras import layers, models
import numpy as np

# Load your data
training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",")

X_train = training_spam[:,1:]
y_train = training_spam[:,0]

X_test = testing_spam[:,1:]
y_test = testing_spam[:,0]

# Normalize your data if necessary
# For example: X_train = tf.keras.utils.normalize(X_train, axis=-1)

# Define your neural network architecture for binary classification
def create_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Use sigmoid for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Use binary_crossentropy for binary classification
    return model

# Create your model
model = create_model()

# Train your model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, class_weight={0:1, 1:2})  # Adjust class_weight based on your data distribution

# Once trained, use your model for predictions
predictions = model.predict(X_test)
# Convert probabilities to binary labels based on a threshold
binary_predictions = (predictions > 0.5).astype(int)
print("Predictions: ", binary_predictions)

# Evaluate your model
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy: ", accuracy)
