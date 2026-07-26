# ==========================================================
# Project: Face Recognition using CNN (LFW Dataset)
# Author: Abhishek Thakur
# ==========================================================

# ==========================
# Import Libraries
# ==========================

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    print("=" * 50)
    print("Loading LFW Dataset...")
    print("=" * 50)

    lfw = fetch_lfw_people(
        min_faces_per_person=50,
        resize=0.5
    )

    X = lfw.images
    y = lfw.target

    class_names = lfw.target_names

    print("\nTotal Images :", X.shape[0])
    print("Image Shape :", X.shape[1:])
    print("Number of Classes :", len(class_names))

    return X, y, class_names


# ==========================================================
# Display Sample Faces
# ==========================================================

def display_faces(X, y, class_names):

    plt.figure(figsize=(12,6))

    for i in range(10):

        plt.subplot(2,5,i+1)

        plt.imshow(X[i], cmap="gray")

        plt.title(class_names[y[i]], fontsize=8)

        plt.axis("off")

    plt.tight_layout()

    plt.show()


# ==========================================================
# Normalize Images
# ==========================================================

def normalize_images(X):

    X = X.astype("float32") / 255.0

    return X


# ==========================================================
# Reshape Images
# ==========================================================

def reshape_images(X):

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        X.shape[2],
        1
    )

    return X


# ==========================================================
# Split Dataset
# ==========================================================

def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    print("\nTraining Images :", X_train.shape)
    print("Testing Images :", X_test.shape)

    return X_train, X_test, y_train, y_test


# ==========================================================
# Build CNN Model
# ==========================================================

def build_model(input_shape, num_classes):

    print("\nBuilding CNN Model...")

    model = Sequential([

        Input(shape=input_shape),

        Conv2D(32, (3,3), activation="relu"),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation="relu"),

        Flatten(),

        Dense(128, activation="relu"),

        Dropout(0.5),

        Dense(num_classes, activation="softmax")

    ])

    model.summary()

    return model


# ==========================================================
# Compile Model
# ==========================================================

def compile_model(model):

    print("\nCompiling Model...")

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    print("Model Compiled Successfully.")


# ==========================================================
# Train CNN Model
# ==========================================================

def train_model(model, X_train, y_train, X_test, y_test):

    print("\nTraining Model...\n")

    history = model.fit(

        X_train,

        y_train,

        epochs=15,

        batch_size=32,

        validation_data=(X_test, y_test),

        verbose=1

    )

    print("\nTraining Completed.")

    return history


# ==========================================================
# Plot Accuracy
# ==========================================================

def plot_accuracy(history):

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training vs Validation Accuracy")

    plt.legend()

    plt.grid(True)

    plt.show()


# ==========================================================
# Plot Loss
# ==========================================================

def plot_loss(history):

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training vs Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.show()

# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(model, X_test, y_test):

    print("\nEvaluating Model...\n")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

    print("\nTest Accuracy : {:.2f}%".format(accuracy * 100))
    print("Test Loss     : {:.4f}".format(loss))


# ==========================================================
# Predict Test Faces
# ==========================================================

def predict_faces(model, X_test):

    print("\nPredicting Test Faces...\n")

    predictions = model.predict(X_test)

    return predictions


# ==========================================================
# Display Prediction Results
# ==========================================================

def display_predictions(predictions, X_test, y_test, class_names):

    plt.figure(figsize=(12,6))

    for i in range(10):

        plt.subplot(2,5,i+1)

        plt.imshow(X_test[i].reshape(X_test.shape[1], X_test.shape[2]), cmap="gray")

        predicted = np.argmax(predictions[i])

        actual = y_test[i]

        plt.title(
            f"P: {class_names[predicted]}\nA: {class_names[actual]}",
            fontsize=8
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()


# ==========================================================
# Predict a Single Face
# ==========================================================

def predict_single_face(model, X_test, y_test, class_names):

    index = 10

    image = np.expand_dims(X_test[index], axis=0)

    prediction = model.predict(image)

    predicted_class = np.argmax(prediction)

    actual_class = y_test[index]

    print("\nSingle Face Prediction")
    print("-" * 40)

    print("Actual Person    :", class_names[actual_class])
    print("Predicted Person :", class_names[predicted_class])

    plt.figure(figsize=(4,4))

    plt.imshow(
        X_test[index].reshape(X_test.shape[1], X_test.shape[2]),
        cmap="gray"
    )

    plt.title(f"Prediction: {class_names[predicted_class]}")

    plt.axis("off")

    plt.show()


# ==========================================================
# Save Trained Model
# ==========================================================

def save_model(model):

    print("\nSaving Model...")

    model.save("face_recognition_lfw_cnn.keras")

    print("Model saved successfully as 'face_recognition_lfw_cnn.keras'")

# ==========================================================
# Main Function
# ==========================================================

def main():

    # Load Dataset
    X, y, class_names = load_dataset()

    # Display Sample Faces
    display_faces(X, y, class_names)

    # Normalize Images
    X = normalize_images(X)

    # Reshape Images
    X = reshape_images(X)

    # Split Dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Build CNN Model
    model = build_model(
        input_shape=(X_train.shape[1], X_train.shape[2], 1),
        num_classes=len(class_names)
    )

    # Compile Model
    compile_model(model)

    # Train Model
    history = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    # Plot Accuracy
    plot_accuracy(history)

    # Plot Loss
    plot_loss(history)

    # Evaluate Model
    evaluate_model(model, X_test, y_test)

    # Predict Faces
    predictions = predict_faces(model, X_test)

    # Display Predictions
    display_predictions(
        predictions,
        X_test,
        y_test,
        class_names
    )

    # Predict Single Face
    predict_single_face(
        model,
        X_test,
        y_test,
        class_names
    )

    # Save Model
    save_model(model)

    print("\n" + "=" * 60)
    print("Face Recognition using CNN (LFW Dataset) Completed Successfully!")
    print("=" * 60)


# ==========================================================
# Execute Program
# ==========================================================

if __name__ == "__main__":
    main()