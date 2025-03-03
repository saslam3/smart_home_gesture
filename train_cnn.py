import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Define CNN Model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(17, activation='softmax')  # 17 gestures
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model (Replace with actual training data)
model = create_model()
# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn.h5")
model.save(MODEL_PATH)
print(f"Model saved as {MODEL_PATH}")
